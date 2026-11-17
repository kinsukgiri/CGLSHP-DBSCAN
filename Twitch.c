#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<omp.h>
#include<time.h>

#define DIM 30
#define MAX_LINE 2000
#define MAX_LANGUAGES 21
#define MAX_LANGUAGE_LENGTH 50
#define UNCLASSIFIED -2
#define NOISE -1
#define MIN_MINPTS 3
#define MAX_MINPTS 30
#define MINPTS_STEP 3
#define EPS_CANDIDATES 8
#define ITERATIONS 15
#define THREAD_COUNT 8
#define GRID_DIMENSIONS 2
#define LSH_HASHES 4
#define LSH_L_COUNT 5
#define RI_THRESHOLD 0.95
#define ECR_MIN_MINPTS 2
#define ECR_MAX_MINPTS 50
#define ECR_PCA_DIM 2
#define ECR_MAX_POWER_ITER 100
#define ECR_EPSILON 1.0e-10
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define LSH_BUCKET_MULTIPLIER 2
#define LSH_WIDTH_FACTOR 1.5

typedef struct 
{
    double value[DIM];
    int cluster;
}point;

typedef struct 
{
    int size,cells;
    int *count;
    int **points;
    double cell_size;
}dynamic_grid;

typedef struct 
{
    double time_sum,silhouette_sum,clusters_sum,noise_sum,distance_sum;
    int validation_success_count;
}result_average;

typedef struct 
{
    int tables,hashes,buckets;
    double width;
    double *random_vectors;
    double *random_offsets;
    int *head;
    int *next;
}lsh_structure;

point *points=NULL;
int n;
int language_count=0;
char language_names[MAX_LANGUAGES][MAX_LANGUAGE_LENGTH];

typedef struct
{
    int *u;
    int *v;
    long long count,capacity;
}edge_buffer;

typedef struct
{
    int threads;
    double time,rand_index,speedup,distance_checks;
}method_result;

typedef struct
{
    int L,M,threads;
    double time,rand_index,speedup,time_score,combined_score,distance_checks;
}lsh_result;

int *csr_row_ptr=NULL;
int *csr_col_idx=NULL;
int csr_nnz=0;
dynamic_grid grid_hybrid;
dynamic_grid grid_a6;
lsh_structure lsh;

void remove_newline(char *text) 
{
    int l=(int)strlen(text);
    while(l>0 && (text[l-1]=='\n'||text[l-1]=='\r')) 
	{
        text[l-1]='\0';
        l--;
    }
}

int get_language_id(char *language) 
{
    int i;
    for(i=0;i<language_count;i++) 
	{
        if(strcmp(language_names[i],language)==0)
            return i;
    }
    if(language_count>=MAX_LANGUAGES)
        return -1;

    strncpy(language_names[language_count],language,MAX_LANGUAGE_LENGTH-1);
    language_names[language_count][MAX_LANGUAGE_LENGTH-1]='\0';
    language_count++;

    return language_count-1;
}

int read_twitch_csv(char *filename) 
{
    FILE *file;
    char line[MAX_LINE];
    int cap=2000;

    file=fopen(filename,"r");
    if(file==NULL) 
	{
        printf("\nerror: cannot open file: %s\n",filename);
        return 0;
    }
    if(fgets(line,MAX_LINE,file)==NULL) 
	{
        fclose(file);
        return 0;
    }
    points=(point *)malloc(sizeof(point)*cap);
    
    if(points==NULL) 
	{
        fclose(file);
        printf("\nerror: memory allocation failed\n");
        return 0;
    }
    
    n=0;
    while(fgets(line,MAX_LINE,file)!=NULL) 
	{
        char temp[11][100];
        char *tok;
        int col,lid,j;

        for(col=0;col<11;col++)
            temp[col][0]='\0';

        tok=strtok(line,",");
        if(tok==NULL)
            continue;

        strncpy(temp[0],tok,99);
        temp[0][99]='\0';

        for(col=1;col<11;col++) 
		{
            tok=strtok(NULL,",");
            if(tok==NULL)
                break;
            strncpy(temp[col],tok,99);
            temp[col][99]='\0';
        }
        if(col<11)
            continue;

        remove_newline(temp[10]);
        points[n].value[0]=atof(temp[1]);
        points[n].value[1]=atof(temp[2]);
        points[n].value[2]=atof(temp[3]);
        points[n].value[3]=atof(temp[4]);
        points[n].value[4]=atof(temp[5]);
        points[n].value[5]=atof(temp[6]);
        points[n].value[6]=atof(temp[7]);

        if(strcmp(temp[8],"TRUE")==0 || strcmp(temp[8],"True")==0 || strcmp(temp[8],"true")==0)
            points[n].value[7]=1.0;
        else
            points[n].value[7]=0.0;

        if(strcmp(temp[9],"TRUE")==0 || strcmp(temp[9],"True")==0 || strcmp(temp[9],"true")==0)
            points[n].value[8]=1.0;
        else
            points[n].value[8]=0.0;

        lid=get_language_id(temp[10]);
        for(j=0;j<MAX_LANGUAGES;j++)
            points[n].value[9+j]=0.0;

        if(lid>=0 && lid<MAX_LANGUAGES)
            points[n].value[9+lid]=1.0;

        points[n].cluster=UNCLASSIFIED;
        n++;

        if(n>=cap) 
		{
            cap=cap*2;
            points=(point *)realloc(points,sizeof(point)*cap);
            if(points==NULL) 
			{
                fclose(file);
                printf("\nerror: memory reallocation failed\n");
                return 0;
            }
        }
    }
    fclose(file);
    return n;
}

void normalize_data() 
{
    double mn[DIM],mx[DIM];
    int d,i;

    if(n<=0)
        return;

    for(d=0;d<9;d++) 
	{
        mn[d]=points[0].value[d];
        mx[d]=points[0].value[d];
    }

    for(i=1;i<n;i++) 
	{
        for(d=0;d<9;d++) 
		{
            if(points[i].value[d]<mn[d])
                mn[d]=points[i].value[d];

            if(points[i].value[d]>mx[d])
                mx[d]=points[i].value[d];
        }
    }
    for(i=0;i<n;i++) 
	{
        for(d=0;d<9;d++) 
		{
            if(mx[d]!=mn[d])
                points[i].value[d]=(points[i].value[d]-mn[d])/(mx[d]-mn[d]);
            else
                points[i].value[d]=0.0;
        }
    }
}

double squared_distance(int a,int b) 
{
    double s=0.0;
    double df;
    int d;

    for(d=0;d<DIM;d++) 
	{
        df=points[a].value[d]-points[b].value[d];
        s+=df*df;
    }
    return s;
}

double euclidean_distance(int a,int b) 
{
    return sqrt(squared_distance(a,b));
}

void reset_clusters() 
{
    int i;
    for(i=0;i<n;i++)
        points[i].cluster=UNCLASSIFIED;
}

void free_csr()
{
    free(csr_row_ptr);
    free(csr_col_idx);
    csr_row_ptr=NULL;
    csr_col_idx=NULL;
    csr_nnz=0;
}

void init_edge_buffers(edge_buffer buffers[],int count)
{
    int i;
    for(i=0;i<count;i++)
    {
        buffers[i].u=NULL;
        buffers[i].v=NULL;
        buffers[i].count=0;
        buffers[i].capacity=0;
    }
}

void free_edge_buffers(edge_buffer buffers[],int count)
{
    int i;
    for(i=0;i<count;i++)
    {
        free(buffers[i].u);
        free(buffers[i].v);
        buffers[i].u=NULL;
        buffers[i].v=NULL;
        buffers[i].count=0;
        buffers[i].capacity=0;
    }
}

int append_edge(edge_buffer *buffer,int u,int v)
{
    long long new_capacity;
    int *new_u;
    int *new_v;

    if(buffer->count>=buffer->capacity)
    {
        if(buffer->capacity==0)
            new_capacity=1024;
        else
            new_capacity=buffer->capacity*2;

        new_u=(int *)malloc(sizeof(int)*(size_t)new_capacity);
        new_v=(int *)malloc(sizeof(int)*(size_t)new_capacity);

        if(new_u==NULL || new_v==NULL)
        {
            free(new_u);
            free(new_v);
            return 0;
        }
        if(buffer->count>0)
        {
            memcpy(new_u,buffer->u,sizeof(int)*(size_t)buffer->count);
            memcpy(new_v,buffer->v,sizeof(int)*(size_t)buffer->count);
        }
        free(buffer->u);
        free(buffer->v);
        buffer->u=new_u;
        buffer->v=new_v;
        buffer->capacity=new_capacity;
    }
    buffer->u[buffer->count]=u;
    buffer->v[buffer->count]=v;
    buffer->count++;
    return 1;
}

int compare_ints(const void *a,const void *b)
{
    int x=*(const int *)a;
    int y=*(const int *)b;
    if(x<y) return -1;
    if(x>y) return 1;
    return 0;
}

int build_csr_from_edges(edge_buffer buffers[],int buffer_count)
{
    int *degree;
    int *cursor;
    int *new_row_ptr;
    int i,t,k;
    long long total=0;

    free_csr();

    degree=(int *)calloc((size_t)n,sizeof(int));
    if(degree==NULL) return 0;

    for(t=0;t<buffer_count;t++)
    {
        for(k=0;k<(int)buffers[t].count;k++)
        {
            int u=buffers[t].u[k];
            int v=buffers[t].v[k];
            degree[u]++;
            if(v!=u) degree[v]++;
        }
    }
    csr_row_ptr=(int *)malloc(sizeof(int)*(size_t)(n+1));
    if(csr_row_ptr==NULL)
    {
        free(degree);
        return 0;
    }
    csr_row_ptr[0]=0;
    for(i=0;i<n;i++)
    {
        csr_row_ptr[i+1]=csr_row_ptr[i]+degree[i];
        total+=degree[i];
    }
    if(total>2147483647LL)
    {
        free(degree);
        free_csr();
        return 0;
    }
    csr_nnz=(int)total;

    if(csr_nnz>0)
    {
        csr_col_idx=(int *)malloc(sizeof(int)*(size_t)csr_nnz);
        if(csr_col_idx==NULL)
        {
            free(degree);
            free_csr();
            return 0;
        }
    }
    cursor=(int *)malloc(sizeof(int)*(size_t)n);
    if(cursor==NULL)
    {
        free(degree);
        free_csr();
        return 0;
    }

    for(i=0;i<n;i++) 
	{
		cursor[i]=csr_row_ptr[i];
	}
    for(t=0;t<buffer_count;t++)
    {
        for(k=0;k<(int)buffers[t].count;k++)
        {
            int u=buffers[t].u[k];
            int v=buffers[t].v[k];
            csr_col_idx[cursor[u]++]=v;
            if(v!=u) csr_col_idx[cursor[v]++]=u;
        }
    }
    new_row_ptr=(int *)malloc(sizeof(int)*(size_t)(n+1));
    if(new_row_ptr==NULL)
    {
        free(cursor);
        free(degree);
        free_csr();
        return 0;
    }
    new_row_ptr[0]=0;
    for(i=0;i<n;i++)
    {
        int begin=csr_row_ptr[i];
        int end=csr_row_ptr[i+1];
        int len=end-begin;
        int write_pos=new_row_ptr[i];
        int j;

        if(len>1)
            qsort(csr_col_idx+begin,(size_t)len,sizeof(int),compare_ints);

        for(j=begin;j<end;j++)
        {
            if(j==begin || csr_col_idx[j]!=csr_col_idx[j-1])
            {
                csr_col_idx[write_pos++]=csr_col_idx[j];
            }
        }
        new_row_ptr[i+1]=write_pos;
    }
    free(csr_row_ptr);
    csr_row_ptr=new_row_ptr;
    csr_nnz=csr_row_ptr[n];

    if(csr_nnz==0)
    {
        free(csr_col_idx);
        csr_col_idx=NULL;
    }
    else
    {
        int *shrunk=(int *)realloc(csr_col_idx,sizeof(int)*(size_t)csr_nnz);
        if(shrunk!=NULL) csr_col_idx=shrunk;
    }
    free(cursor);
    free(degree);
    return 1;
}

int csr_region_query(int p,int nei[])
{
    int c=0;
    int pos;
    if(csr_row_ptr==NULL) return 0;
    for(pos=csr_row_ptr[p];pos<csr_row_ptr[p+1];pos++)
        nei[c++]=csr_col_idx[pos];
    return c;
}

int sequential_region_query(int p,double eps,int nei[],long long *dc) 
{
    double e2=eps*eps;
    int c=0;
    int i;

    for(i=0;i<n;i++) 
	{
        if(squared_distance(p,i)<=e2) 
		{
            nei[c]=i;
            c++;
        }
        (*dc)++;
    }
    return c;
}

void add_seed(int p,int seed[],int *sc,int added[]) 
{
    if(p<0 || p>=n)
        return;

    if(added[p]!=0)
        return;

    if(*sc>=n)
        return;

    seed[*sc]=p;
    (*sc)++;
    added[p]=1;
}

long long sequential_dbscan(double eps,int minpts) 
{
    int *nei;
    int *seed;
    int *added;
    int cid=0;
    int i;
    long long dc=0;

    nei=(int *)malloc(sizeof(int)*n);
    seed=(int *)malloc(sizeof(int)*n);
    added=(int *)malloc(sizeof(int)*n);

    if(nei==NULL || seed==NULL || added==NULL) 
	{
        free(nei);
        free(seed);
        free(added);
        return 0;
    }
    reset_clusters();

    for(i=0;i<n;i++) 
	{
        int nc,pos;
        int sc=0;

        if(points[i].cluster!=UNCLASSIFIED)
            continue;

        nc=sequential_region_query(i,eps,nei,&dc);
        if(nc<minpts) 
		{
            points[i].cluster=NOISE;
            continue;
        }
        points[i].cluster=cid;
        memset(added,0,sizeof(int)*n);

        for(pos=0;pos<nc;pos++) 
		{
            if(nei[pos]!=i)
                add_seed(nei[pos],seed,&sc,added);
        }
        pos=0;
        while(pos<sc) 
		{
            int cur,nc2,j;
            cur=seed[pos];

            if(points[cur].cluster==NOISE)
                points[cur].cluster=cid;

            if(points[cur].cluster!=UNCLASSIFIED) 
			{
                pos++;
                continue;
            }
            points[cur].cluster=cid;
            nc2=sequential_region_query(cur,eps,nei,&dc);

            if(nc2>=minpts) 
			{
                for(j=0;j<nc2;j++) 
				{
                    int q=nei[j];
                    if(points[q].cluster==UNCLASSIFIED)
                        add_seed(q,seed,&sc,added);
                    else if(points[q].cluster==NOISE)
                        points[q].cluster=cid;
                }
            }
            pos++;
        }
        cid++;
    }
    free(nei);
    free(seed);
    free(added);
    return dc;
}

long long build_neighbors_a1(double eps,int threads)
{
    double e2=eps*eps;
    long long dc=0;
    int i;
    edge_buffer *buffers;
    int ok=1;

    buffers=(edge_buffer *)malloc(sizeof(edge_buffer)*(size_t)threads);
    if(buffers==NULL) return 0;
    init_edge_buffers(buffers,threads);
    omp_set_num_threads(threads);

    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        long long lc=0;

        #pragma omp for schedule(static)
        for(i=0;i<n;i++)
        {
            int j;
            for(j=i;j<n;j++)
            {
                lc++;
                if(squared_distance(i,j)<=e2)
                {
                    if(!append_edge(&buffers[tid],i,j)) ok=0;
                }
            }
        }
        #pragma omp atomic
        dc+=lc;
    }
    if(!ok || !build_csr_from_edges(buffers,threads))
    {
        free_edge_buffers(buffers,threads);
        free(buffers);
        free_csr();
        return 0;
    }
    free_edge_buffers(buffers,threads);
    free(buffers);
    return dc;
}

void dbscan_from_csr(int minpts)
{
    int *nei;
    int *seed;
    int *added;
    int cid=0;
    int i;

    nei=(int *)malloc(sizeof(int)*n);
    seed=(int *)malloc(sizeof(int)*n);
    added=(int *)malloc(sizeof(int)*n);

    if(nei==NULL || seed==NULL || added==NULL)
    {
        free(nei); free(seed); free(added);
        return;
    }
    reset_clusters();

    for(i=0;i<n;i++)
    {
        int nc,sc=0,pos;
        if(points[i].cluster!=UNCLASSIFIED) continue;

        nc=csr_region_query(i,nei);
        if(nc<minpts)
        {
            points[i].cluster=NOISE;
            continue;
        }
        points[i].cluster=cid;
        memset(added,0,sizeof(int)*n);

        for(pos=0;pos<nc;pos++)
            if(nei[pos]!=i) add_seed(nei[pos],seed,&sc,added);

        pos=0;
        while(pos<sc)
        {
            int cur=seed[pos];
            int nc2,j;

            if(points[cur].cluster==NOISE)
                points[cur].cluster=cid;

            if(points[cur].cluster!=UNCLASSIFIED)
            {
                pos++;
                continue;
            }
            points[cur].cluster=cid;
            nc2=csr_region_query(cur,nei);

            if(nc2>=minpts)
            {
                for(j=0;j<nc2;j++)
                {
                    int q=nei[j];
                    if(points[q].cluster==UNCLASSIFIED)
                        add_seed(q,seed,&sc,added);
                    else if(points[q].cluster==NOISE)
                        points[q].cluster=cid;
                }
            }
            pos++;
        }
        cid++;
    }
    free(nei); free(seed); free(added);
}

long long dbscan_a1(double eps,int minpts,int threads)
{
    long long dc=build_neighbors_a1(eps,threads);
    dbscan_from_csr(minpts);
    return dc;
}

int allocate_dynamic_grid(dynamic_grid *grid,int size,double cell_size) 
{
    int i;

    grid->size=size;
    grid->cells=size*size;
    grid->cell_size=cell_size;

    grid->count=(int *)calloc(grid->cells,sizeof(int));

    if(grid->count==NULL)
        return 0;

    grid->points=(int **)malloc(sizeof(int *)*grid->cells);
    if(grid->points==NULL) 
	{
        free(grid->count);
        grid->count=NULL;
        return 0;
    }
    for(i=0;i<grid->cells;i++) 
	{
        grid->points[i]=(int *)malloc(sizeof(int)*n);
        if(grid->points[i]==NULL) 
		{
            int j;
            for(j=0;j<i;j++)
                free(grid->points[j]);

            free(grid->points);
            free(grid->count);

            grid->points=NULL;
            grid->count=NULL;

            return 0;
        }
    }
    return 1;
}

void clear_dynamic_grid(dynamic_grid *grid) 
{
    int i;
    for(i=0;i<grid->cells;i++)
        grid->count[i]=0;
}

int get_dynamic_grid_cell(dynamic_grid *grid,int p) 
{
    int x,y;

    x=(int)(points[p].value[0]/grid->cell_size);
    y=(int)(points[p].value[1]/grid->cell_size);

    if(x<0)
        x=0;

    if(x>=grid->size)
        x=grid->size-1;

    if(y<0)
        y=0;

    if(y>=grid->size)
        y=grid->size-1;

    return y*grid->size+x;
}

void build_dynamic_grid(dynamic_grid *grid)
{
    int i,cell;
    clear_dynamic_grid(grid);

    for(i=0;i<n;i++) 
	{
        cell=get_dynamic_grid_cell(grid,i);
        grid->points[cell][grid->count[cell]]=i;
        grid->count[cell]++;
    }
}

double minimum_distance_to_cell(int p,int cx,int cy,double cs) 
{
    double px,py,cminx,cmaxx,cminy,cmaxy;
    double dx=0.0;
    double dy=0.0;

    px=points[p].value[0];
    py=points[p].value[1];

    cminx=cx*cs;
    cmaxx=(cx+1)*cs;
    cminy=cy*cs;
    cmaxy=(cy+1)*cs;

    if(px<cminx)
        dx=cminx-px;
    else if(px>cmaxx)
        dx=px-cmaxx;

    if(py<cminy)
        dy=cminy-py;
    else if(py>cmaxy)
        dy=py-cmaxy;

    return sqrt(dx*dx+dy*dy);
}

long long build_neighbors_hybrid(dynamic_grid *grid,double eps,int threads)
{
    double e2=eps*eps;
    long long dc=0;
    int radius;
    int i;
    edge_buffer *buffers;
    int ok=1;

    buffers=(edge_buffer *)malloc(sizeof(edge_buffer)*(size_t)threads);
    if(buffers==NULL) return 0;
    init_edge_buffers(buffers,threads);
    build_dynamic_grid(grid);

    radius=(int)ceil(eps/grid->cell_size);
    if(radius<1) radius=1;
    if(radius>grid->size) radius=grid->size;

    omp_set_num_threads(threads);

    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        long long lc=0;

        #pragma omp for schedule(dynamic)
        for(i=0;i<n;i++)
        {
            int cell=get_dynamic_grid_cell(grid,i);
            int cx=cell%grid->size;
            int cy=cell/grid->size;
            int minx=cx-radius,maxx=cx+radius;
            int miny=cy-radius,maxy=cy+radius;
            int x,y;

            if(minx<0) minx=0;
            if(maxx>=grid->size) maxx=grid->size-1;
            if(miny<0) miny=0;
            if(maxy>=grid->size) maxy=grid->size-1;

            for(x=minx;x<=maxx;x++)
            {
                for(y=miny;y<=maxy;y++)
                {
                    int nc,k;
                    if(minimum_distance_to_cell(i,x,y,grid->cell_size)>eps)
                        continue;

                    nc=y*grid->size+x;

                    for(k=0;k<grid->count[nc];k++)
                    {
                        int j=grid->points[nc][k];
                        if(j<i) 
							continue;

                        lc++;
                        if(squared_distance(i,j)<=e2)
                        {
                            if(!append_edge(&buffers[tid],i,j)) ok=0;
                        }
                    }
                }
            }
        }
        #pragma omp atomic
        dc+=lc;
    }
    if(!ok || !build_csr_from_edges(buffers,threads))
    {
        free_edge_buffers(buffers,threads);
        free(buffers);
        free_csr();
        return 0;
    }
    free_edge_buffers(buffers,threads);
    free(buffers);
    return dc;
}

long long dbscan_a3(double eps,int minpts,int threads)
{
    long long dc=build_neighbors_hybrid(&grid_hybrid,eps,threads);
    dbscan_from_csr(minpts);
    return dc;
}

double lsh_uniform() 
{
    return ((double)rand()+1.0)/((double)RAND_MAX+2.0);
}

double lsh_gaussian() 
{
    double u1,u2;

    u1=lsh_uniform();
    u2=lsh_uniform();

    return sqrt(-2.0*log(u1))*cos(2.0*M_PI*u2);
}

void free_lsh() 
{
    if(lsh.random_vectors!=NULL) 
	{
        free(lsh.random_vectors);
        lsh.random_vectors=NULL;
    }
    if(lsh.random_offsets!=NULL) 
	{
        free(lsh.random_offsets);
        lsh.random_offsets=NULL;
    }
    if(lsh.head!=NULL) 
	{
        free(lsh.head);
        lsh.head=NULL;
    }
    if(lsh.next!=NULL) 
	{
        free(lsh.next);
        lsh.next=NULL;
    }
    lsh.tables=0;
    lsh.hashes=0;
    lsh.buckets=0;
    lsh.width=0.0;
}

unsigned long long lsh_hash_value(int table,int point_id,int hash_id)
{
    double projection=0.0;
    int d;
    for(d=0;d<DIM;d++)
        projection+=points[point_id].value[d]*lsh.random_vectors[((table*lsh.hashes+hash_id)*DIM)+d];

    projection+=lsh.random_offsets[table*lsh.hashes+hash_id];
    projection=floor(projection/lsh.width);

    {
        long long bucket_value=(long long)projection;
        unsigned long long x=(unsigned long long)bucket_value;
        x^=x>>33;
        x*=0xff51afd7ed558ccdULL;
        x^=x>>33;
        x*=0xc4ceb9fe1a85ec53ULL;
        x^=x>>33;
        return x;
    }
}

int get_lsh_bucket(int table,int point_id)
{
    unsigned long long combined=0;
    int h;
    for(h=0;h<lsh.hashes;h++)
    {
        unsigned long long hv=lsh_hash_value(table,point_id,h);
        combined^=hv+0x9e3779b97f4a7c15ULL+(combined<<6)+(combined>>2);
    }
    return (int)(combined%(unsigned long long)lsh.buckets);
}

int initialize_lsh(double eps,int tables,int hashes)
{
    size_t random_vector_size,random_offset_size,head_size,next_size;
    int i,t,b;

    free_lsh();
    lsh.tables=tables;
    lsh.hashes=hashes;
    lsh.width=LSH_WIDTH_FACTOR*eps;
    if(lsh.width<=0.0) lsh.width=0.001;

    lsh.buckets=n*LSH_BUCKET_MULTIPLIER;
    if(lsh.buckets<1000) lsh.buckets=1000;

    random_vector_size=(size_t)lsh.tables*(size_t)lsh.hashes*(size_t)DIM;
    random_offset_size=(size_t)lsh.tables*(size_t)lsh.hashes;
    head_size=(size_t)lsh.tables*(size_t)lsh.buckets;
    next_size=(size_t)lsh.tables*(size_t)n;

    lsh.random_vectors=(double *)malloc(sizeof(double)*random_vector_size);
    lsh.random_offsets=(double *)malloc(sizeof(double)*random_offset_size);
    lsh.head=(int *)malloc(sizeof(int)*head_size);
    lsh.next=(int *)malloc(sizeof(int)*next_size);

    if(lsh.random_vectors==NULL || lsh.random_offsets==NULL || lsh.head==NULL || lsh.next==NULL)
    {
        printf("\nerror: LSH memory allocation failed.");
        free_lsh();
        return 0;
    }
    for(t=0;t<lsh.tables;t++)
        for(b=0;b<lsh.buckets;b++)
            lsh.head[t*lsh.buckets+b]=-1;

    for(i=0;i<(int)random_vector_size;i++) 
	{
		lsh.random_vectors[i]=lsh_gaussian();
	}
    for(i=0;i<(int)random_offset_size;i++) 
	{
		lsh.random_offsets[i]=lsh_uniform()*lsh.width;
	}
    for(i=0;i<n;i++)
    {
        for(t=0;t<lsh.tables;t++)
        {
            int bucket=get_lsh_bucket(t,i);
            int index=t*lsh.buckets+bucket;
            lsh.next[t*n+i]=lsh.head[index];
            lsh.head[index]=i;
        }
    }
    return 1;
}

long long build_neighbors_lsh(double eps,int threads,int tables,int hashes)
{
    double e2=eps*eps;
    long long dc=0;
    int i;
    edge_buffer *buffers;
    int ok=1;

    buffers=(edge_buffer *)malloc(sizeof(edge_buffer)*(size_t)threads);
    if(buffers==NULL) return 0;
    init_edge_buffers(buffers,threads);

    if(!initialize_lsh(eps,tables,hashes))
    {
        free(buffers);
        return 0;
    }

    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        long long lc=0;

        #pragma omp for schedule(dynamic)
        for(i=0;i<n;i++)
        {
            int t;
            for(t=0;t<lsh.tables;t++)
            {
                int bucket=get_lsh_bucket(t,i);
                int index=t*lsh.buckets+bucket;
                int q=lsh.head[index];

                while(q!=-1)
                {
                    if(q>=i)
                    {
                        lc++;
                        if(squared_distance(i,q)<=e2)
                        {
                            if(!append_edge(&buffers[tid],i,q)) ok=0;
                        }
                    }
                    q=lsh.next[t*n+q];
                }
            }
        }
        #pragma omp atomic
        dc+=lc;
    }
    if(!ok || !build_csr_from_edges(buffers,threads))
    {
        free_edge_buffers(buffers,threads);
        free(buffers);
        free_csr();
        return 0;
    }
    free_edge_buffers(buffers,threads);
    free(buffers);
    return dc;
}

long long dbscan_a4(double eps,int minpts,int threads,int tables,int hashes)
{
    long long dc=build_neighbors_lsh(eps,threads,tables,hashes);
    dbscan_from_csr(minpts);
    return dc;
}

int count_clusters() 
{
    int *used;
    int c=0;
    int i;

    used=(int *)calloc(n,sizeof(int));
    if(used==NULL)
        return 0;

    for(i=0;i<n;i++) 
	{
        if(points[i].cluster>=0 && points[i].cluster<n)
            used[points[i].cluster]=1;
    }
    for(i=0;i<n;i++) 
	{
        if(used[i]!=0)
            c++;
    }
    free(used);
    return c;
}

int count_noise() 
{
    int c=0;
    int i;

    for(i=0;i<n;i++) 
	{
        if(points[i].cluster==NOISE)
            c++;
    }
    return c;
}

int validate_labels_against_sequential(int seq[]) 
{
    int *p2s;
    int *s2p;
    int i;

    p2s=(int *)malloc(sizeof(int)*n);
    s2p=(int *)malloc(sizeof(int)*n);

    if(p2s==NULL || s2p==NULL) 
	{
        free(p2s);
        free(s2p);
        return 0;
    }
    for(i=0;i<n;i++) 
	{
        p2s[i]=UNCLASSIFIED;
        s2p[i]=UNCLASSIFIED;
    }
    for(i=0;i<n;i++) 
	{
        int sc=seq[i];
        int pc=points[i].cluster;

        if(sc==NOISE || pc==NOISE) 
		{
            if(sc!=pc) 
			{
                free(p2s);
                free(s2p);
                return 0;
            }
            continue;
        }
        if(sc<0 || pc<0) 
		{
            free(p2s);
            free(s2p);
            return 0;
        }
        if(p2s[pc]==UNCLASSIFIED)
            p2s[pc]=sc;
        else if(p2s[pc]!=sc) 
		{
            free(p2s);
            free(s2p);
            return 0;
        }
        if(s2p[sc]==UNCLASSIFIED)
            s2p[sc]=pc;
        else if(s2p[sc]!=pc) 
		{
            free(p2s);
            free(s2p);
            return 0;
        }
    }
    free(p2s);
    free(s2p);
    return 1;
}

double calculate_rand_index(int reference[],int predicted[])
{
    long long agree=0;
    long long total=0;
    int i,j;

    for(i=0;i<n-1;i++)
    {
        for(j=i+1;j<n;j++)
        {
            int r_same=(reference[i]==reference[j]);
            int p_same=(predicted[i]==predicted[j]);
            if(r_same==p_same) agree++;
            total++;
        }
    }
    if(total==0) return 0.0;
    return (double)agree/(double)total;
}

double calculate_silhouette() 
{
    double total=0.0;
    int valid=0;
    int cc,i;

    cc=count_clusters();
    if(cc<=1)
        return -1.0;

    for(i=0;i<n;i++) 
	{
        int own,oc2,j;
        int oc=0;
        double a=0.0;
        double b=10000.0;

        if(points[i].cluster<0)
            continue;

        own=points[i].cluster;
        for(j=0;j<n;j++) 
		{
            if(j!=i && points[j].cluster==own) 
			{
                a+=euclidean_distance(i,j);
                oc++;
            }
        }
        if(oc==0)
            continue;

        a/=oc;
        for(oc2=0;oc2<n;oc2++) 
		{
            double ds=0.0;
            int cnt=0;

            if(oc2==own)
                continue;

            for(j=0;j<n;j++) 
			{
                if(points[j].cluster==oc2) 
				{
                    ds+=euclidean_distance(i,j);
                    cnt++;
                }
            }
            if(cnt>0 && ds/cnt<b)
                b=ds/cnt;
        }
        if(b==10000.0)
            continue;

        if(a<b)
            total+=(b-a)/b;
        else if(a>b)
            total+=(b-a)/a;

        valid++;
    }
    if(valid==0)
        return -1.0;

    return total/valid;
}

double calculate_k_distance(int p,int k) 
{
    double *ds;
    int c=0;
    int i,j;

    ds=(double *)malloc(sizeof(double)*n);

    if(ds==NULL)
        return 0.0;

    for(i=0;i<n;i++) 
	{
        if(i!=p) 
		{
            ds[c]=euclidean_distance(p,i);
            c++;
        }
    }
    for(i=0;i<c-1;i++) 
	{
        for(j=i+1;j<c;j++) 
		{
            if(ds[j]<ds[i]) 
			{
                double t=ds[i];
                ds[i]=ds[j];
                ds[j]=t;
            }
        }
    }
    if(c==0) 
	{
        free(ds);
        return 0.0;
    }
    if(k<=0)
        k=1;

    if(k>c)
        k=c;

    {
        double r=ds[k-1];
        free(ds);
        return r;
    }
}

int generate_eps_candidates(int minpts,double eps_values[]) 
{
    double *ds;
    int pct[EPS_CANDIDATES]={50,60,70,80,85,90,95,98};
    int i,j;

    ds=(double *)malloc(sizeof(double)*n);

    if(ds==NULL)
        return 0;

    for(i=0;i<n;i++)
        ds[i]=calculate_k_distance(i,minpts);

    for(i=0;i<n-1;i++) 
	{
        for(j=i+1;j<n;j++) 
		{
            if(ds[j]<ds[i]) 
			{
                double t=ds[i];
                ds[i]=ds[j];
                ds[j]=t;
            }
        }
    }
    for(i=0;i<EPS_CANDIDATES;i++) 
	{
        int idx=(n*pct[i])/100;

        if(idx>=n)
            idx=n-1;

        eps_values[i]=ds[idx];
    }
    free(ds);
    return EPS_CANDIDATES;
}

void optimize_parameters(double *be,int *bm,double *bs) 
{
    double bsl=-1.0;
    double bel=0.0;
    int bml=0;
    int minpts;

    for(minpts=MIN_MINPTS;minpts<=MAX_MINPTS;minpts+=MINPTS_STEP) 
	{
        double ev[EPS_CANDIDATES];
        int cnt,c;

        cnt=generate_eps_candidates(minpts,ev);
        printf("\n\nTesting minpts = %d",minpts);

        for(c=0;c<cnt;c++) 
		{
            double eps,sil;
            long long dc;
            int cl,noise;

            eps=ev[c];
            dc=sequential_dbscan(eps,minpts);
            cl=count_clusters();
            noise=count_noise();
            sil=calculate_silhouette();

            printf("\n  eps = %.6f clusters = %d noise = %d silhouette = %.6f",eps,cl,noise,sil);

            if(cl<2)
                continue;

            if(noise>=(int)(0.90*n))
                continue;

            if(sil<0.0)
                continue;

            if(sil>bsl) 
			{
                bsl=sil;
                bel=eps;
                bml=minpts;
            }
            (void)dc;
        }
    }
    if(bml==0) 
	{
        bml=5;
        bel=0.30;
        bsl=-1.0;
    }
    *be=bel;
    *bm=bml;
    *bs=bsl;
}

typedef struct
{
    int a,b,c;
}ecr_triangle;

typedef struct
{
    int a,b;
}ecr_edge;

typedef struct
{
    double x,y;
}ecr_xy;

int ecr_edge_equal(ecr_edge e1,ecr_edge e2)
{
    return ((e1.a==e2.a && e1.b==e2.b) || (e1.a==e2.b && e1.b==e2.a));
}

void ecr_make_edge(ecr_edge *e,int a,int b)
{
    if(a<b)
    {
        e->a=a;
        e->b=b;
    }
    else
    {
        e->a=b;
        e->b=a;
    }
}

double ecr_cross(ecr_xy a,ecr_xy b,ecr_xy c)
{
    return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
}

void ecr_swap_xy(ecr_xy *a,ecr_xy *b)
{
    ecr_xy t=*a;
    *a=*b;
    *b=t;
}

int ecr_point_compare(ecr_xy a,ecr_xy b)
{
    if(a.x<b.x) return -1;
    if(a.x>b.x) return 1;
    if(a.y<b.y) return -1;
    if(a.y>b.y) return 1;
    return 0;
}

void ecr_sort_points(ecr_xy p[],int order[],int count)
{
    int i,j;
    for(i=0;i<count;i++)
        order[i]=i;

    for(i=0;i<count-1;i++)
    {
        for(j=i+1;j<count;j++)
        {
            if(ecr_point_compare(p[order[j]],p[order[i]])<0)
            {
                int t=order[i];
                order[i]=order[j];
                order[j]=t;
            }
        }
    }
}

int ecr_convex_hull(ecr_xy p[],int count,int hull[])
{
    int *order;
    int *h;
    int m=0;
    int i;

    if(count<3)
        return 0;

    order=(int *)malloc(sizeof(int)*count);
    h=(int *)malloc(sizeof(int)*(2*count+1));

    if(order==NULL || h==NULL)
    {
        free(order);
        free(h);
        return 0;
    }
    ecr_sort_points(p,order,count);
    for(i=0;i<count;i++)
    {
        int id=order[i];
        while(m>=2 && ecr_cross(p[h[m-2]],p[h[m-1]],p[id])<=ECR_EPSILON)
            m--;
        h[m++]=id;
    }
    {
        int lower=m;
        for(i=count-2;i>=0;i--)
        {
            int id=order[i];
            while(m>lower && ecr_cross(p[h[m-2]],p[h[m-1]],p[id])<=ECR_EPSILON)
                m--;
            h[m++]=id;
        }
    }
    if(m>1)
        m--;

    for(i=0;i<m;i++)
        hull[i]=h[i];

    free(order);
    free(h);
    return m;
}

int ecr_point_in_hull(ecr_xy p,ecr_xy data[],int hull[],int hcount)
{
    int i;
    int sign=0;

    if(hcount<3)
        return 0;

    for(i=0;i<hcount;i++)
    {
        ecr_xy a=data[hull[i]];
        ecr_xy b=data[hull[(i+1)%hcount]];
        double cr=ecr_cross(a,b,p);
        if(cr>ECR_EPSILON)
            sign=1;
        else if(cr<-ECR_EPSILON)
            return 0;
    }
    return sign>=0;
}

int ecr_circumcircle(ecr_xy a,ecr_xy b,ecr_xy c,double *ux,double *uy,double *r2)
{
    double d;
    double ax=a.x,ay=a.y;
    double bx=b.x,by=b.y;
    double cx=c.x,cy=c.y;
    double aa=ax*ax+ay*ay;
    double bb=bx*bx+by*by;
    double cc=cx*cx+cy*cy;

    d=2.0*(ax*(by-cy)+bx*(cy-ay)+cx*(ay-by));

    if(fabs(d)<ECR_EPSILON)
        return 0;

    *ux=(aa*(by-cy)+bb*(cy-ay)+cc*(ay-by))/d;
    *uy=(aa*(cx-bx)+bb*(ax-cx)+cc*(bx-ax))/d;

    *r2=(*ux-ax)*(*ux-ax)+(*uy-ay)*(*uy-ay);
    return 1;
}

int ecr_point_in_circumcircle(ecr_xy p,ecr_triangle t,ecr_xy data[])
{
    double ux,uy,r2,ds;

    if(!ecr_circumcircle(data[t.a],data[t.b],data[t.c],&ux,&uy,&r2))
        return 0;

    ds=(p.x-ux)*(p.x-ux)+(p.y-uy)*(p.y-uy);
    return ds<=r2+1.0e-9;
}

void ecr_orient_triangle(ecr_triangle *t,ecr_xy data[])
{
    if(ecr_cross(data[t->a],data[t->b],data[t->c])<0.0)
    {
        int x=t->b;
        t->b=t->c;
        t->c=x;
    }
}

int ecr_add_triangle(ecr_triangle **triangles,int *count,int *capacity,int a,int b,int c,ecr_xy data[])
{
    ecr_triangle t;
    if(*count>=*capacity)
    {
        int newcap=(*capacity)*2;
        ecr_triangle *tmp=(ecr_triangle *)realloc(*triangles,sizeof(ecr_triangle)*newcap);
        if(tmp==NULL)
            return 0;
        *triangles=tmp;
        *capacity=newcap;
    }
    t.a=a;
    t.b=b;
    t.c=c;
    ecr_orient_triangle(&t,data);
    (*triangles)[*count]=t;
    (*count)++;
    return 1;
}

int ecr_delaunay(ecr_xy original[],int count,ecr_triangle **out_triangles,int *out_count)
{
    ecr_xy *data;
    ecr_triangle *triangles;
    int tri_count=0;
    int tri_capacity=64;
    int *bad;
    int bad_capacity;
    ecr_edge *edges;
    int edge_count,edge_capacity,hull_count;
    double minx,maxx,miny,maxy,span;
    int i,j;

    *out_triangles=NULL;
    *out_count=0;

    if(count<3)
        return 0;

    data=(ecr_xy *)malloc(sizeof(ecr_xy)*(count+3));
    triangles=(ecr_triangle *)malloc(sizeof(ecr_triangle)*tri_capacity);
    bad_capacity=tri_capacity;
    bad=(int *)malloc(sizeof(int)*bad_capacity);
    edges=(ecr_edge *)malloc(sizeof(ecr_edge)*128);

    if(data==NULL || triangles==NULL || bad==NULL || edges==NULL)
    {
        free(data); free(triangles); free(bad); free(edges);
        return 0;
    }
    for(i=0;i<count;i++)
        data[i]=original[i];

    minx=maxx=data[0].x;
    miny=maxy=data[0].y;

    for(i=1;i<count;i++)
    {
        if(data[i].x<minx) minx=data[i].x;
        if(data[i].x>maxx) maxx=data[i].x;
        if(data[i].y<miny) miny=data[i].y;
        if(data[i].y>maxy) maxy=data[i].y;
    }
    span=maxx-minx;
    if(maxy-miny>span) span=maxy-miny;
    if(span<ECR_EPSILON) span=1.0;

    data[count].x=(minx+maxx)/2.0;
    data[count].y=maxy+20.0*span;
    data[count+1].x=minx-20.0*span;
    data[count+1].y=miny-5.0*span;
    data[count+2].x=maxx+20.0*span;
    data[count+2].y=miny-5.0*span;

    if(!ecr_add_triangle(&triangles,&tri_count,&tri_capacity,count,count+1,count+2,data))
    {
        free(data); free(triangles); free(bad); free(edges);
        return 0;
    }
    for(i=0;i<count;i++)
    {
        int bad_count=0;
        edge_count=0;
        edge_capacity=128;

        for(j=0;j<tri_count;j++)
        {
            if(ecr_point_in_circumcircle(data[i],triangles[j],data))
            {
                if(bad_count>=bad_capacity)
                {
                    int newcap=bad_capacity*2;
                    int *tmp=(int *)realloc(bad,sizeof(int)*newcap);
                    if(tmp==NULL)
                    {
                        free(data); free(triangles); free(bad); free(edges);
                        return 0;
                    }
                    bad=tmp;
                    bad_capacity=newcap;
                }
                bad[bad_count++]=j;

                if(edge_count+3>=edge_capacity)
                {
                    int newcap=edge_capacity*2;
                    while(newcap<=edge_count+3) newcap*=2;
                    ecr_edge *tmp=(ecr_edge *)realloc(edges,sizeof(ecr_edge)*newcap);
                    if(tmp==NULL)
                    {
                        free(data); free(triangles); free(bad); free(edges);
                        return 0;
                    }
                    edges=tmp;
                    edge_capacity=newcap;
                }
                ecr_make_edge(&edges[edge_count++],triangles[j].a,triangles[j].b);
                ecr_make_edge(&edges[edge_count++],triangles[j].b,triangles[j].c);
                ecr_make_edge(&edges[edge_count++],triangles[j].c,triangles[j].a);
            }
        }
        if(bad_count>0)
        {
            int write=0;
            for(j=0;j<tri_count;j++)
            {
                int k,remove=0;
                for(k=0;k<bad_count;k++)
                {
                    if(bad[k]==j)
                    {
                        remove=1;
                        break;
                    }
                }
                if(!remove)
                    triangles[write++]=triangles[j];
            }
            tri_count=write;
        }
        for(j=0;j<edge_count;j++)
        {
            int k,occ=1;
            for(k=0;k<edge_count;k++)
            {
                if(k!=j && ecr_edge_equal(edges[j],edges[k]))
                    occ++;
            }
            if(occ==1)
            {
                if(!ecr_add_triangle(&triangles,&tri_count,&tri_capacity,
                                     edges[j].a,edges[j].b,i,data))
                {
                    free(data); free(triangles); free(bad); free(edges);
                    return 0;
                }
            }
        }
    }
    {
        int write=0;
        for(i=0;i<tri_count;i++)
        {
            ecr_triangle t=triangles[i];
            if(t.a>=count || t.b>=count || t.c>=count)
                continue;
            triangles[write++]=t;
        }
        tri_count=write;
    }
    {
        int *hull=(int *)malloc(sizeof(int)*count);
        if(hull!=NULL)
            hull_count=ecr_convex_hull(original,count,hull);
        else
            hull_count=0;

        if(hull_count>=3)
        {
            int write=0;
            for(i=0;i<tri_count;i++)
            {
                double ux,uy,r2;
                ecr_triangle t=triangles[i];

                if(!ecr_circumcircle(original[t.a],original[t.b],original[t.c],&ux,&uy,&r2))
                    continue;

                {
                    ecr_xy center;
                    center.x=ux;
                    center.y=uy;
                    if(ecr_point_in_hull(center,original,hull,hull_count))
                        triangles[write++]=t;
                }
            }
            tri_count=write;
        }
        free(hull);
    }
    free(data);
    free(bad);
    free(edges);

    *out_triangles=triangles;
    *out_count=tri_count;
    return 1;
}

void ecr_sort_radii(double radii[],int count)
{
    int i,j;
    for(i=0;i<count-1;i++)
    {
        for(j=i+1;j<count;j++)
        {
            if(radii[j]<radii[i])
            {
                double t=radii[i];
                radii[i]=radii[j];
                radii[j]=t;
            }
        }
    }
}

double ecr_knee_from_radii(double radii[],int count)
{
    int i,best=0;
    double x1,y1,x2,y2,den,best_dist=-1.0;

    if(count<=0)
        return 0.0;
    if(count==1)
        return radii[0];

    x1=0.0;
    y1=radii[0];
    x2=(double)(count-1);
    y2=radii[count-1];
    den=sqrt((y1-y2)*(y1-y2)+(x2-x1)*(x2-x1));

    if(den<ECR_EPSILON)
        return radii[count/2];

    for(i=0;i<count;i++)
    {
        double x=(double)i;
        double y=radii[i];
        double dist=fabs((y1-y2)*x+(x2-x1)*y+x1*y2-x2*y1)/den;

        if(dist>best_dist)
        {
            best_dist=dist;
            best=i;
        }
    }
    return radii[best];
}

int ecr_pca_to_2d(double out[][ECR_PCA_DIM])
{
    double mean[DIM];
    double cov[DIM][DIM];
    double v1[DIM],v2[DIM],tmp[DIM];
    int i,j,k,it;

    if(n<3)
        return 0;

    for(j=0;j<DIM;j++)
        mean[j]=0.0;

    for(i=0;i<n;i++)
        for(j=0;j<DIM;j++)
            mean[j]+=points[i].value[j];

    for(j=0;j<DIM;j++)
        mean[j]/=(double)n;

    for(i=0;i<DIM;i++)
        for(j=0;j<DIM;j++)
            cov[i][j]=0.0;

    for(i=0;i<DIM;i++)
    {
        for(j=i;j<DIM;j++)
        {
            double s=0.0;
            for(k=0;k<n;k++)
                s+=(points[k].value[i]-mean[i])*(points[k].value[j]-mean[j]);
            cov[i][j]=s/(double)(n-1);
            cov[j][i]=cov[i][j];
        }
    }
    for(j=0;j<DIM;j++)
        v1[j]=1.0/(double)(j+1);

    {
        double norm=0.0;
        for(j=0;j<DIM;j++) norm+=v1[j]*v1[j];
        norm=sqrt(norm);
        for(j=0;j<DIM;j++) v1[j]/=norm;
    }
    for(it=0;it<ECR_MAX_POWER_ITER;it++)
    {
        double norm=0.0;
        for(i=0;i<DIM;i++)
        {
            tmp[i]=0.0;
            for(j=0;j<DIM;j++)
                tmp[i]+=cov[i][j]*v1[j];
        }
        for(i=0;i<DIM;i++) norm+=tmp[i]*tmp[i];
        norm=sqrt(norm);
        if(norm<ECR_EPSILON) break;
        for(i=0;i<DIM;i++) tmp[i]/=norm;
        for(i=0;i<DIM;i++) v1[i]=tmp[i];
    }
    for(j=0;j<DIM;j++)
        v2[j]=1.0/(double)(DIM-j+1);

    {
        double dot=0.0,norm=0.0;
        for(j=0;j<DIM;j++) dot+=v2[j]*v1[j];
        for(j=0;j<DIM;j++) v2[j]-=dot*v1[j];
        for(j=0;j<DIM;j++) norm+=v2[j]*v2[j];
        norm=sqrt(norm);
        if(norm<ECR_EPSILON) norm=1.0;
        for(j=0;j<DIM;j++) v2[j]/=norm;
    }
    for(it=0;it<ECR_MAX_POWER_ITER;it++)
    {
        double norm=0.0,dot;
        for(i=0;i<DIM;i++)
        {
            tmp[i]=0.0;
            for(j=0;j<DIM;j++)
                tmp[i]+=cov[i][j]*v2[j];
        }
        dot=0.0;
        for(i=0;i<DIM;i++) dot+=tmp[i]*v1[i];
        for(i=0;i<DIM;i++) tmp[i]-=dot*v1[i];
        for(i=0;i<DIM;i++) norm+=tmp[i]*tmp[i];
        norm=sqrt(norm);
        if(norm<ECR_EPSILON) break;
        for(i=0;i<DIM;i++) v2[i]=tmp[i]/norm;
    }
    for(i=0;i<n;i++)
    {
        out[i][0]=0.0;
        out[i][1]=0.0;
        for(j=0;j<DIM;j++)
        {
            double z=points[i].value[j]-mean[j];
            out[i][0]+=z*v1[j];
            out[i][1]+=z*v2[j];
        }
    }
    return 1;
}

double ecr_distance_2d(double data[][ECR_PCA_DIM],int a,int b)
{
    double dx=data[a][0]-data[b][0];
    double dy=data[a][1]-data[b][1];
    return sqrt(dx*dx+dy*dy);
}

void ecr_reset_labels(int labels[])
{
    int i;
    for(i=0;i<n;i++)
        labels[i]=UNCLASSIFIED;
}

int ecr_region_query(double data[][ECR_PCA_DIM],int p,double eps,int neighbors[])
{
    int c=0,i;
    for(i=0;i<n;i++)
    {
        if(ecr_distance_2d(data,p,i)<=eps)
            neighbors[c++]=i;
    }
    return c;
}

void ecr_dbscan_2d(double data[][ECR_PCA_DIM],double eps,int minpts,int labels[])
{
    int *neighbors=(int *)malloc(sizeof(int)*n);
    int *seed=(int *)malloc(sizeof(int)*n);
    int *added=(int *)malloc(sizeof(int)*n);
    int cid=0,i;

    if(neighbors==NULL || seed==NULL || added==NULL)
    {
        free(neighbors); free(seed); free(added);
        ecr_reset_labels(labels);
        return;
    }
    ecr_reset_labels(labels);

    for(i=0;i<n;i++)
    {
        int nc,pos,sc;

        if(labels[i]!=UNCLASSIFIED)
            continue;

        nc=ecr_region_query(data,i,eps,neighbors);
        if(nc<minpts)
        {
            labels[i]=NOISE;
            continue;
        }
        labels[i]=cid;
        memset(added,0,sizeof(int)*n);
        sc=0;

        for(pos=0;pos<nc;pos++)
        {
            int q=neighbors[pos];
            if(q!=i && !added[q])
            {
                seed[sc++]=q;
                added[q]=1;
            }
        }
        pos=0;
        while(pos<sc)
        {
            int cur=seed[pos++];
            int nc2;
            int j;

            if(labels[cur]==NOISE)
                labels[cur]=cid;

            if(labels[cur]!=UNCLASSIFIED)
                continue;

            labels[cur]=cid;
            nc2=ecr_region_query(data,cur,eps,neighbors);

            if(nc2>=minpts)
            {
                for(j=0;j<nc2;j++)
                {
                    int q=neighbors[j];
                    if(q!=cur && !added[q] && sc<n)
                    {
                        seed[sc++]=q;
                        added[q]=1;
                    }
                }
            }
        }
        cid++;
    }
    free(neighbors);
    free(seed);
    free(added);
}

int ecr_count_clusters(int labels[])
{
    int *used=(int *)calloc(n,sizeof(int));
    int c=0,i;
    if(used==NULL) return 0;
    for(i=0;i<n;i++)
        if(labels[i]>=0 && labels[i]<n)
            used[labels[i]]=1;
    for(i=0;i<n;i++)
        if(used[i]) c++;
    free(used);
    return c;
}

int ecr_count_noise(int labels[])
{
    int c=0,i;
    for(i=0;i<n;i++)
        if(labels[i]==NOISE) c++;
    return c;
}

double ecr_silhouette(double data[][ECR_PCA_DIM],int labels[])
{
    double total=0.0;
    int valid=0;
    int cc=ecr_count_clusters(labels);
    int i,j,c;

    if(cc<=1)
        return -1.0;

    for(i=0;i<n;i++)
    {
        int own=labels[i];
        int own_count=0;
        double a=0.0;
        double b=1.0e100;

        if(own<0)
            continue;

        for(j=0;j<n;j++)
        {
            if(j!=i && labels[j]==own)
            {
                a+=ecr_distance_2d(data,i,j);
                own_count++;
            }
        }
        if(own_count==0)
            continue;
        a/=(double)own_count;

        for(c=0;c<n;c++)
        {
            int count=0;
            double sum=0.0;
            if(c==own) continue;

            for(j=0;j<n;j++)
            {
                if(labels[j]==c)
                {
                    sum+=ecr_distance_2d(data,i,j);
                    count++;
                }
            }
            if(count>0 && sum/(double)count<b)
                b=sum/(double)count;
        }
        if(b>=1.0e99)
            continue;

        if(a<b)
            total+=(b-a)/b;
        else if(a>b)
            total+=(b-a)/a;

        valid++;
    }
    if(valid==0)
        return -1.0;
    return total/(double)valid;
}

int ecr_select_parameters(double *eps,int *minpts,double *silhouette,double *selection_time,int *ec_count)
{
    double (*pca)[ECR_PCA_DIM];
    ecr_triangle *triangles=NULL;
    int triangle_count=0;
    double *radii=NULL;
    int radius_count=0;
    int *labels=NULL;
    double best_sil=-1.0;
    int best_mp=0;
    double start=omp_get_wtime();
    int i;

    pca=(double (*)[ECR_PCA_DIM])malloc(sizeof(double[ECR_PCA_DIM])*n);
    labels=(int *)malloc(sizeof(int)*n);

    if(pca==NULL || labels==NULL)
    {
        free(pca); free(labels);
        return 0;
    }

    printf("\n\n===============================================");
    printf("\n        A5 ECR PARAMETER SELECTION");
    printf("\n===============================================");

    if(!ecr_pca_to_2d(pca))
    {
        free(pca); free(labels);
        return 0;
    }
    if(!ecr_delaunay((ecr_xy *)pca,n,&triangles,&triangle_count))
    {
        free(pca); free(labels);
        return 0;
    }
    if(triangle_count>0)
        radii=(double *)malloc(sizeof(double)*triangle_count);

    if(radii==NULL && triangle_count>0)
    {
        free(pca); free(labels); free(triangles);
        return 0;
    }
    for(i=0;i<triangle_count;i++)
    {
        double ux,uy,r2;
        ecr_triangle t=triangles[i];
        if(ecr_circumcircle(((ecr_xy *)pca)[t.a],
                            ((ecr_xy *)pca)[t.b],
                            ((ecr_xy *)pca)[t.c],&ux,&uy,&r2))
            radii[radius_count++]=sqrt(r2);
    }
    if(radius_count>0)
        ecr_sort_radii(radii,radius_count);

    *ec_count=radius_count;
    if(radius_count==0)
    {
        *eps=0.0;
        *minpts=5;
        *silhouette=-1.0;
        *selection_time=omp_get_wtime()-start;
        free(pca); free(labels); free(triangles); free(radii);
        return 1;
    }
    *eps=ecr_knee_from_radii(radii,radius_count);
    printf("\nnumber of Delaunay triangles retained : %d",triangle_count);
    printf("\nnumber of empty-circle radii        : %d",radius_count);
    printf("\nECR knee epsilon in PCA-2D          : %.10f",*eps);
    printf("\n\nMinPts search: 2 to 50");

    for(i=ECR_MIN_MINPTS;i<=ECR_MAX_MINPTS;i++)
    {
        int clusters,noise;
        double sil;

        ecr_dbscan_2d(pca,*eps,i,labels);
        clusters=ecr_count_clusters(labels);
        noise=ecr_count_noise(labels);
        sil=ecr_silhouette(pca,labels);

        if(clusters>=2 && noise<(int)(0.90*n) && sil>=0.0)
        {
            printf("\n  MinPts=%d clusters=%d noise=%d silhouette=%.6f",
                   i,clusters,noise,sil);

            if(sil>best_sil)
            {
                best_sil=sil;
                best_mp=i;
            }
        }
    }
    if(best_mp==0)
    {
        best_mp=5;
        best_sil=-1.0;
    }
    *minpts=best_mp;
    *silhouette=best_sil;
    *selection_time=omp_get_wtime()-start;

    printf("\n\nECR selected eps     = %.10f",*eps);
    printf("\nECR selected MinPts  = %d",*minpts);
    printf("\nECR selection silhouette = %.6f",*silhouette);
    printf("\nECR selection time   = %.6f seconds",*selection_time);

    free(pca);
    free(labels);
    free(triangles);
    free(radii);
    return 1;
}

typedef struct
{
    int start_cell;
    int end_cell;
    long long cost;
}a6_partition;

int *a6_parent=NULL;
unsigned char *a6_core=NULL;

int a6_find_root(int x)
{
    int p=x;
    int r;

    while(a6_parent[p]!=p)
        p=a6_parent[p];

    r=p;
    while(a6_parent[x]!=x)
    {
        int next=a6_parent[x];
        a6_parent[x]=r;
        x=next;
    }

    return r;
}

void a6_union(int a,int b)
{
    int ra,rb,small,big,old;

    ra=a6_find_root(a);
    rb=a6_find_root(b);

    while(ra!=rb)
    {
        if(ra<rb)
        {
            small=ra;
            big=rb;
        }
        else
        {
            small=rb;
            big=ra;
        }
        old=__sync_val_compare_and_swap(&a6_parent[big],big,small);

        if(old==big)
            break;

        ra=a6_find_root(a);
        rb=a6_find_root(b);
    }
}

void free_a6_union_find()
{
    free(a6_parent);
    free(a6_core);
    a6_parent=NULL;
    a6_core=NULL;
}

long long a6_cell_neighbor_cost(dynamic_grid *grid,int cell)
{
    int cx=cell%grid->size;
    int cy=cell/grid->size;
    int x,y;
    long long neighbor_points=0;

    for(y=cy-1;y<=cy+1;y++)
    {
        for(x=cx-1;x<=cx+1;x++)
        {
            int nc;

            if(x<0 || x>=grid->size || y<0 || y>=grid->size)
                continue;

            nc=y*grid->size+x;
            neighbor_points+=(long long)grid->count[nc];
        }
    }
    return (long long)grid->count[cell]*neighbor_points;
}

int a6_make_partitions(dynamic_grid *grid,int threads,a6_partition part[])
{
    long long *costs;
    long long total_cost=0;
    long long target;
    long long accumulated=0;
    int cell=0;
    int t;

    costs=(long long *)malloc(sizeof(long long)*grid->cells);

    if(costs==NULL)
        return 0;

    for(cell=0;cell<grid->cells;cell++)
    {
        costs[cell]=a6_cell_neighbor_cost(grid,cell);
        total_cost+=costs[cell];
    }
    if(total_cost<=0)
    {
        for(t=0;t<threads;t++)
        {
            part[t].start_cell=(grid->cells*t)/threads;
            part[t].end_cell=(grid->cells*(t+1))/threads;
            part[t].cost=0;
        }

        free(costs);
        return 1;
    }
    target=(total_cost+threads-1)/threads;
    cell=0;

    for(t=0;t<threads;t++)
    {
        long long local=0;
        int start=cell;

        if(t==threads-1)
            cell=grid->cells;

        while(cell<grid->cells)
        {
            long long next_cost=local+costs[cell];

            if(t<threads-1 && cell>start && next_cost>target)
                break;

            local=next_cost;
            cell++;

            if(t<threads-1 && local>=target)
                break;
        }
        part[t].start_cell=start;
        part[t].end_cell=cell;
        part[t].cost=local;

        accumulated+=local;
    }
    part[threads-1].end_cell=grid->cells;

    if(part[threads-1].end_cell<part[threads-1].start_cell)
        part[threads-1].end_cell=part[threads-1].start_cell;

    free(costs);
    (void)accumulated;
    return 1;
}

long long a6_build_neighbors(dynamic_grid *grid,double eps,int threads)
{
    double e2=eps*eps;
    edge_buffer *buffers;
    a6_partition *part;
    long long dc=0;
    int t;
    int ok=1;

    buffers=(edge_buffer *)malloc(sizeof(edge_buffer)*(size_t)threads);
    part=(a6_partition *)malloc(sizeof(a6_partition)*(size_t)threads);

    if(buffers==NULL || part==NULL)
    {
        free(buffers);
        free(part);
        return 0;
    }
    init_edge_buffers(buffers,threads);
    build_dynamic_grid(grid);

    if(!a6_make_partitions(grid,threads,part))
    {
        free_edge_buffers(buffers,threads);
        free(buffers);
        free(part);
        return 0;
    }
    omp_set_num_threads(threads);
    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        long long local_dc=0;
        int cell;

        for(cell=part[tid].start_cell;cell<part[tid].end_cell;cell++)
        {
            int cx=cell%grid->size;
            int cy=cell/grid->size;
            int x,y;

            if(grid->count[cell]==0)
                continue;

            {
                int sk;
                for(sk=0;sk<grid->count[cell];sk++)
                {
                    int i=grid->points[cell][sk];
                    for(y=cy-1;y<=cy+1;y++)
                    {
                        for(x=cx-1;x<=cx+1;x++)
                        {
                            int nc,nk;

                            if(x<0 || x>=grid->size || y<0 || y>=grid->size)
                                continue;

                            nc=y*grid->size+x;
                            for(nk=0;nk<grid->count[nc];nk++)
                            {
                                int j=grid->points[nc][nk];
                                if(j<i)
                                    continue;

                                local_dc++;
                                if(squared_distance(i,j)<=e2)
                                {
                                    if(!append_edge(&buffers[tid],i,j))
                                        ok=0;
                                }
                            }
                        }
                    }
                }
            }
        }
        #pragma omp atomic
        dc+=local_dc;
    }
    if(!ok || !build_csr_from_edges(buffers,threads))
    {
        free_edge_buffers(buffers,threads);
        free(buffers);
        free(part);
        free_csr();
        return 0;
    }
    free_edge_buffers(buffers,threads);
    free(buffers);
    free(part);

    return dc;
}

void a6_parallel_cluster_from_csr(int minpts,int threads)
{
    int *root_to_cluster;
    int *cluster_root;
    int *cluster_id;
    int i,pos;
    int cluster_count=0;

    free_a6_union_find();

    a6_parent=(int *)malloc(sizeof(int)*(size_t)n);
    a6_core=(unsigned char *)calloc((size_t)n,sizeof(unsigned char));

    if(a6_parent==NULL || a6_core==NULL)
    {
        free_a6_union_find();
        reset_clusters();
        return;
    }
    for(i=0;i<n;i++)
        a6_parent[i]=i;

    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(dynamic)
    for(i=0;i<n;i++)
    {
        int degree=csr_row_ptr[i+1]-csr_row_ptr[i];
        if(degree>=minpts)
            a6_core[i]=1;
    }

    #pragma omp parallel for schedule(dynamic)
    for(i=0;i<n;i++)
    {
        if(a6_core[i])
        {
            int ppos;
            for(ppos=csr_row_ptr[i];ppos<csr_row_ptr[i+1];ppos++)
            {
                int q=csr_col_idx[ppos];

                if(q>i && a6_core[q])
                    a6_union(i,q);
            }
        }
    }
    root_to_cluster=(int *)malloc(sizeof(int)*(size_t)n);
    cluster_root=(int *)malloc(sizeof(int)*(size_t)n);
    cluster_id=(int *)malloc(sizeof(int)*(size_t)n);

    if(root_to_cluster==NULL || cluster_root==NULL || cluster_id==NULL)
    {
        free(root_to_cluster);
        free(cluster_root);
        free(cluster_id);
        reset_clusters();
        free_a6_union_find();
        return;
    }
    for(i=0;i<n;i++)
        root_to_cluster[i]=-1;

    for(i=0;i<n;i++)
    {
        if(a6_core[i])
        {
            int root=a6_find_root(i);
            if(root_to_cluster[root]<0)
            {
                root_to_cluster[root]=cluster_count;
                cluster_root[cluster_count]=root;
                cluster_id[cluster_count]=cluster_count;
                cluster_count++;
            }
        }
    }
    
    #pragma omp parallel for schedule(static)
    for(i=0;i<n;i++)
    {
        if(a6_core[i])
        {
            int root=a6_find_root(i);
            points[i].cluster=root_to_cluster[root];
        }
        else
        {
            int best_root=n+1;
            int ppos;
            for(ppos=csr_row_ptr[i];ppos<csr_row_ptr[i+1];ppos++)
            {
                int q=csr_col_idx[ppos];
                if(a6_core[q])
                {
                    int root=a6_find_root(q);
                    if(root<best_root)
                        best_root=root;
                }
            }
            if(best_root<=n)
                points[i].cluster=root_to_cluster[best_root];
            else
                points[i].cluster=NOISE;
        }
    }
    free(root_to_cluster);
    free(cluster_root);
    free(cluster_id);
}

long long dbscan_a6(double eps,int minpts,int threads)
{
    long long dc;
    dc=a6_build_neighbors(&grid_a6,eps,threads);

    if(csr_row_ptr!=NULL)
        a6_parallel_cluster_from_csr(minpts,threads);
    else
        reset_clusters();

    return dc;
}

void free_dynamic_grid(dynamic_grid *grid) 
{
    int i;
    if(grid->points!=NULL) 
	{
        for(i=0;i<grid->cells;i++)
            free(grid->points[i]);

        free(grid->points);
        grid->points=NULL;
    }
    if(grid->count!=NULL) 
	{
        free(grid->count);
        grid->count=NULL;
    }
    grid->size=0;
    grid->cells=0;
    grid->cell_size=0.0;
}

void free_memory() 
{
    if(points!=NULL) 
	{
        free(points);
        points=NULL;
    }
    free_csr();
    free_dynamic_grid(&grid_hybrid);
    free_dynamic_grid(&grid_a6);
    free_lsh();
}


double label_entropy(const int labels[],int count)
{
    int *values=(int *)malloc(sizeof(int)*count);
    int *freq=(int *)calloc((size_t)count,sizeof(int));
    int unique=0,i,j;
    double h=0.0;
    if(values==NULL || freq==NULL) { free(values); free(freq); return 0.0; }
    for(i=0;i<count;i++)
    {
        for(j=0;j<unique;j++) if(values[j]==labels[i]) break;
        if(j==unique) { values[unique]=labels[i]; freq[unique]=0; unique++; }
        freq[j]++;
    }
    for(i=0;i<unique;i++)
    {
        double p=(double)freq[i]/count;
        if(p>0.0) h-=p*log(p);
    }
    free(values); free(freq);
    return h;
}

void calculate_cm_hm(const int reference[],const int predicted[],double *cm,double *hm)
{
    int *rv=(int *)malloc(sizeof(int)*n);
    int *pv=(int *)malloc(sizeof(int)*n);
    int *rf=(int *)calloc((size_t)n,sizeof(int));
    int *pf=(int *)calloc((size_t)n,sizeof(int));
    int *joint=(int *)calloc((size_t)n*(size_t)n,sizeof(int));
    int nr=0,np=0,i,j;
    double hr=0.0,hp=0.0,h_pred_given_ref=0.0,h_ref_given_pred=0.0;
    if(!rv || !pv || !rf || !pf || !joint)
    {
        free(rv); free(pv); free(rf); free(pf); free(joint);
        *cm=0.0; *hm=0.0; return;
    }
    for(i=0;i<n;i++)
    {
        for(j=0;j<nr;j++) if(rv[j]==reference[i]) break;
        if(j==nr) rv[nr++]=reference[i];
        for(j=0;j<np;j++) if(pv[j]==predicted[i]) break;
        if(j==np) pv[np++]=predicted[i];
    }
    for(i=0;i<n;i++)
    {
        int ri=0,pi=0;
        for(j=0;j<nr;j++) if(rv[j]==reference[i]) {ri=j;break;}
        for(j=0;j<np;j++) if(pv[j]==predicted[i]) {pi=j;break;}
        rf[ri]++; pf[pi]++; joint[ri*np+pi]++;
    }
    for(i=0;i<nr;i++) { double p=(double)rf[i]/n; if(p>0) hr-=p*log(p); }
    for(i=0;i<np;i++) { double p=(double)pf[i]/n; if(p>0) hp-=p*log(p); }
    for(i=0;i<nr;i++) for(j=0;j<np;j++) if(joint[i*np+j]>0)
    {
        double p=(double)joint[i*np+j]/n;
        h_pred_given_ref-=p*log((double)joint[i*np+j]/rf[i]);
        h_ref_given_pred-=p*log((double)joint[i*np+j]/pf[j]);
    }
    /*
       Completeness = 1 - H(predicted | reference) / H(predicted).
       Homogeneity  = 1 - H(reference | predicted) / H(reference).
       Here reference is the sequential DBSCAN labeling and predicted is
       the labeling produced by the method being evaluated.
    */
    *cm=(hp<=1.0e-15)?1.0:1.0-h_pred_given_ref/hp;
    *hm=(hr<=1.0e-15)?1.0:1.0-h_ref_given_pred/hr;
    if(*cm<0) *cm=0; if(*cm>1) *cm=1;
    if(*hm<0) *hm=0; if(*hm>1) *hm=1;
    free(rv); free(pv); free(rf); free(pf); free(joint);
}

int main()
{
    char filename[300];
    double best_eps,best_score;
    int best_minpts;
    double a5_eps,a5_sil,a5_selection_time=0.0;
    int a5_minpts,a5_ec_count=0;
    int threads[THREAD_COUNT]={2,4,8,10,16,32,64,88};
    int l_values[LSH_L_COUNT]={2,4,6,8,10};
    int M=4;
    int i,iteration;
    int *seq_labels;
    double a0_time=0.0;
    double a0_sil=0.0;
    int a0_clusters=0,a0_noise=0;
    long long a0_distances=0;
    double iteration_speedup[6][ITERATIONS];

    method_result a1_best={0,1.0e100,0.0,0.0,0.0};
    method_result a3_best={0,1.0e100,0.0,0.0,0.0};
    method_result a6_best={0,1.0e100,0.0,0.0,0.0};
    lsh_result a4_results[LSH_L_COUNT];
    lsh_result a5_results[LSH_L_COUNT];
    int a4_best_index=0;
    int a5_best_index=0;

    printf("\n==========================================");
    printf("\n\nA0 = Sequential DBSCAN");
    printf("\nA1 = Redundancy-Reduced OpenMP DBSCAN");
    printf("\nA3 = Hybrid Grid + OpenMP DBSCAN");
    printf("\nA4 = LSH + OpenMP DBSCAN");
    printf("\nA5 = ECR parameter selection + A4 LSH");
    printf("\nA6 = HPDBSCAN");
    printf("\n\nenter filename: ");
    scanf("%299s",filename);

    n=read_twitch_csv(filename);
    if(n<=0)
    {
        printf("\nerror: dataset could not be loaded.\n");
        free_memory();
        return 1;
    }

    printf("\n\ndataset loaded successfully.");
    printf("\nrows               : %d",n);
    printf("\ndropped column     : channel");
    printf("\nlanguages encoded  : %d",language_count);

    normalize_data();
    printf("\nnormalization      : completed");
    printf("\n\n=========================================");
    printf("\nEXISTING AUTOMATIC PARAMETER SELECTION");
    printf("\n=========================================");
    printf("\n\nminpts search:");
    printf("\n3 6 9 12 15 18 21 24 27 30");
    printf("\neps candidates are generated from k-distance values.");

    optimize_parameters(&best_eps,&best_minpts,&best_score);

    printf("\n\n==================================");
    printf("\nEXISTING SELECTED PARAMETERS");
    printf("\n==================================");
    printf("\n\neps        = %.8f",best_eps);
    printf("\nminpts     = %d",best_minpts);
    printf("\nsilhouette = %.6f",best_score);

    {
        double cs=best_eps/sqrt((double)DIM);
        int gs=(int)ceil(1.0/cs);
        if(gs<1) gs=1;

        printf("\n\n==================================");
        printf("\n A3 GRID CONFIGURATION");
        printf("\n==================================");
        printf("\n\ndimensions d       : %d",DIM);
        printf("\ngrid cell size     : %.10f",cs);
        printf("\ngrid size          : %d x %d",gs,gs);
        printf("\nnumber of cells    : %d",gs*gs);

        if(!allocate_dynamic_grid(&grid_hybrid,gs,cs))
        {
            printf("\nerror: A3 grid memory allocation failed.\n");
            free_memory();
            return 1;
        }
    }

    {
        double a6_cs=best_eps;
        int a6_gs=(int)ceil(1.0/a6_cs);

        if(a6_cs<=0.0)
            a6_cs=0.001;

        if(a6_gs<1) a6_gs=1;

        printf("\n\n==================================");
        printf("\n A6 GRID CONFIGURATION");
        printf("\n==================================");
        printf("\n\nepsilon-sized cell : %.10f",a6_cs);
        printf("\ngrid size           : %d x %d",a6_gs,a6_gs);
        printf("\nnumber of cells     : %d",a6_gs*a6_gs);

        if(!allocate_dynamic_grid(&grid_a6,a6_gs,a6_cs))
        {
            printf("\nerror: A6 grid memory allocation failed.\n");
            free_memory();
            return 1;
        }
    }
    printf("\n\n==================================");
    printf("\n A4 LSH CONFIGURATION");
    printf("\n==================================");
    printf("\n\nM hashes per table : %d",M);
    printf("\nL values            : 2 4 6 8 10");
    printf("\nLSH width factor    : %.2f",LSH_WIDTH_FACTOR);
    printf("\nLSH bucket multiplier: %d",LSH_BUCKET_MULTIPLIER);

    if(!ecr_select_parameters(&a5_eps,&a5_minpts,&a5_sil,&a5_selection_time,&a5_ec_count))
    {
        printf("\nerror: A5 ECR parameter selection failed.\n");
        free_memory();
        return 1;
    }
    printf("\n\n==================================");
    printf("\n A5 SELECTED PARAMETERS");
    printf("\n==================================");
    printf("\n\neps        = %.10f",a5_eps);
    printf("\nminpts     = %d",a5_minpts);
    printf("\nECR silhouette = %.6f",a5_sil);
    printf("\nempty-circle radii = %d",a5_ec_count);

    seq_labels=(int *)malloc(sizeof(int)*n);
    if(seq_labels==NULL)
    {
        printf("\nerror: A0 label memory allocation failed.\n");
        free_memory();
        return 1;
    }
    printf("\n\n==================================");
    printf("\n A0 REFERENCE");
    printf("\n==================================");

    for(iteration=0;iteration<ITERATIONS;iteration++)
    {
        double start=omp_get_wtime();
        long long dc=sequential_dbscan(best_eps,best_minpts);
        double end=omp_get_wtime();
        a0_time+=end-start;
        a0_distances+=dc;

        if(iteration==0)
            for(i=0;i<n;i++) seq_labels[i]=points[i].cluster;

        a0_sil+=calculate_silhouette();
        a0_clusters+=count_clusters();
        a0_noise+=count_noise();
    }

    a0_time/=ITERATIONS;
    a0_sil/=ITERATIONS;
    a0_clusters/=ITERATIONS;
    a0_noise/=ITERATIONS;
    a0_distances/=ITERATIONS;

    printf("\nA0 average time = %.6f seconds",a0_time);
    printf("\nA0 Rand Index   = 1.000000 (reference)");
    printf("\n\n==================================");
    printf("\n       A1 THREAD EXPERIMENT");
    printf("\n==================================");

    {
        method_result r[THREAD_COUNT];
        for(i=0;i<THREAD_COUNT;i++)
        {
            r[i].threads=threads[i];
            r[i].time=0.0;
            r[i].rand_index=0.0;
            r[i].distance_checks=0.0;
        }

        for(iteration=0;iteration<ITERATIONS;iteration++)
        {
            for(i=0;i<THREAD_COUNT;i++)
            {
                int *predicted=(int *)malloc(sizeof(int)*n);
                int j;
                double start,end,ri;
                long long dc;

                if(predicted==NULL)
                {
                    printf("\nerror: A1 label allocation failed.\n");
                    free(seq_labels); free_memory(); return 1;
                }

                start=omp_get_wtime();
                dc=dbscan_a1(best_eps,best_minpts,threads[i]);
                end=omp_get_wtime();

                for(j=0;j<n;j++) predicted[j]=points[j].cluster;
                ri=calculate_rand_index(seq_labels,predicted);

                r[i].time+=end-start;
                r[i].rand_index+=ri;
                r[i].distance_checks+=(double)dc;
                free(predicted);
            }
        }

        for(i=0;i<THREAD_COUNT;i++)
        {
            r[i].time/=ITERATIONS;
            r[i].rand_index/=ITERATIONS;
            r[i].distance_checks/=ITERATIONS;
            r[i].speedup=a0_time/r[i].time;
            printf("\nA1 threads=%d time=%.6f RI=%.6f",threads[i],r[i].time,r[i].rand_index);

            if(r[i].rand_index>=RI_THRESHOLD)
            {
                if(r[i].time<a1_best.time)
                    a1_best=r[i];
            }
        }

        if(a1_best.time>=1.0e99)
        {
            a1_best=r[0];
            for(i=1;i<THREAD_COUNT;i++)
            {
                if(r[i].rand_index>a1_best.rand_index ||
                  (r[i].rand_index==a1_best.rand_index && r[i].time<a1_best.time))
                    a1_best=r[i];
            }
        }
    }
    printf("\n\n==================================");
    printf("\n       A3 THREAD EXPERIMENT");
    printf("\n==================================");

    {
        method_result r[THREAD_COUNT];
        for(i=0;i<THREAD_COUNT;i++)
        {
            r[i].threads=threads[i];
            r[i].time=0.0;
            r[i].rand_index=0.0;
            r[i].distance_checks=0.0;
        }

        for(iteration=0;iteration<ITERATIONS;iteration++)
        {
            for(i=0;i<THREAD_COUNT;i++)
            {
                int *predicted=(int *)malloc(sizeof(int)*n);
                int j;
                double start,end,ri;
                long long dc;

                if(predicted==NULL)
                {
                    printf("\nerror: A3 label allocation failed.\n");
                    free(seq_labels); free_memory(); return 1;
                }

                start=omp_get_wtime();
                dc=dbscan_a3(best_eps,best_minpts,threads[i]);
                end=omp_get_wtime();

                for(j=0;j<n;j++) predicted[j]=points[j].cluster;
                ri=calculate_rand_index(seq_labels,predicted);

                r[i].time+=end-start;
                r[i].rand_index+=ri;
                r[i].distance_checks+=(double)dc;
                free(predicted);
            }
        }

        for(i=0;i<THREAD_COUNT;i++)
        {
            r[i].time/=ITERATIONS;
            r[i].rand_index/=ITERATIONS;
            r[i].distance_checks/=ITERATIONS;
            r[i].speedup=a0_time/r[i].time;
            printf("\nA3 threads=%d time=%.6f RI=%.6f",threads[i],r[i].time,r[i].rand_index);

            if(r[i].rand_index>=RI_THRESHOLD)
            {
                if(r[i].time<a3_best.time)
                    a3_best=r[i];
            }
        }

        if(a3_best.time>=1.0e99)
        {
            a3_best=r[0];
            for(i=1;i<THREAD_COUNT;i++)
            {
                if(r[i].rand_index>a3_best.rand_index ||
                  (r[i].rand_index==a3_best.rand_index && r[i].time<a3_best.time))
                    a3_best=r[i];
            }
        }
    }
    printf("\n\n==============================================");
    printf("\n A4 LSH PARAMETER EXPERIMENT");
    printf("\n==============================================");

    for(i=0;i<LSH_L_COUNT;i++)
    {
        double total_time[THREAD_COUNT];
        double total_ri[THREAD_COUNT];
        double total_dc[THREAD_COUNT];
        int ti;

        a4_results[i].L=l_values[i];
        a4_results[i].M=M;

        for(ti=0;ti<THREAD_COUNT;ti++)
        {
            total_time[ti]=0.0;
            total_ri[ti]=0.0;
            total_dc[ti]=0.0;
        }

        for(iteration=0;iteration<ITERATIONS;iteration++)
        {
            for(ti=0;ti<THREAD_COUNT;ti++)
            {
                int *predicted=(int *)malloc(sizeof(int)*n);
                int j;
                double start,end,ri;
                long long dc;

                if(predicted==NULL)
                {
                    printf("\nerror: A4 label allocation failed.\n");
                    free(seq_labels); free_memory(); return 1;
                }

                srand(10);
                start=omp_get_wtime();
                dc=dbscan_a4(best_eps,best_minpts,threads[ti],l_values[i],M);
                end=omp_get_wtime();

                for(j=0;j<n;j++) predicted[j]=points[j].cluster;
                ri=calculate_rand_index(seq_labels,predicted);

                total_time[ti]+=end-start;
                total_ri[ti]+=ri;
                total_dc[ti]+=(double)dc;
                free(predicted);
            }
        }

        {
            int found=0;
            double best_time_local=1.0e100;
            int best_ti=0;

            for(ti=0;ti<THREAD_COUNT;ti++)
            {
                double avg_time=total_time[ti]/ITERATIONS;
                double avg_ri=total_ri[ti]/ITERATIONS;

                if(avg_ri>=RI_THRESHOLD && avg_time<best_time_local)
                {
                    best_time_local=avg_time;
                    best_ti=ti;
                    found=1;
                }
            }

            if(!found)
            {
                best_ti=0;
                for(ti=1;ti<THREAD_COUNT;ti++)
                {
                    double ri1=total_ri[ti]/ITERATIONS;
                    double ri2=total_ri[best_ti]/ITERATIONS;
                    double time1=total_time[ti]/ITERATIONS;
                    double time2=total_time[best_ti]/ITERATIONS;

                    if(ri1>ri2 || (ri1==ri2 && time1<time2))
                        best_ti=ti;
                }
            }

            a4_results[i].threads=threads[best_ti];
            a4_results[i].time=total_time[best_ti]/ITERATIONS;
            a4_results[i].rand_index=total_ri[best_ti]/ITERATIONS;
            a4_results[i].distance_checks=total_dc[best_ti]/ITERATIONS;
            a4_results[i].speedup=a0_time/a4_results[i].time;
        }

        printf("\nA4 L=%d M=%d best_threads=%d time=%.6f RI=%.6f",
               l_values[i],M,a4_results[i].threads,
               a4_results[i].time,a4_results[i].rand_index);
    }

    a4_best_index=-1;
    for(i=0;i<LSH_L_COUNT;i++)
    {
        if(a4_results[i].rand_index>=RI_THRESHOLD)
        {
            if(a4_best_index==-1 || a4_results[i].time<a4_results[a4_best_index].time)
                a4_best_index=i;
        }
    }

    if(a4_best_index==-1)
    {
        a4_best_index=0;
        for(i=1;i<LSH_L_COUNT;i++)
        {
            if(a4_results[i].rand_index>a4_results[a4_best_index].rand_index ||
              (a4_results[i].rand_index==a4_results[a4_best_index].rand_index &&
               a4_results[i].time<a4_results[a4_best_index].time))
                a4_best_index=i;
        }
    }

    printf("\n\n================================================================================");
    printf("\n  A4 LSH PARAMETER RESULTS");
    printf("\n================================================================================");
    printf("\n\n%-5s%-5s%-9s%-14s%-12s%-12s%-14s", "L","M","Threads","Avg Time","Speedup","Rand Index","Distance Checks");
    printf("\n--------------------------------------------------------------------------------");

    for(i=0;i<LSH_L_COUNT;i++)
    {
        printf("\n%-5d%-5d%-9d%-14.6f%-12.4f%-12.6f%-14.0f",
               a4_results[i].L,a4_results[i].M,a4_results[i].threads,
               a4_results[i].time,a4_results[i].speedup,
               a4_results[i].rand_index,a4_results[i].distance_checks);
    }

    printf("\n\nOptimal A4 configuration:");
    printf("\nL = %d, M = %d, threads = %d",
           a4_results[a4_best_index].L,
           a4_results[a4_best_index].M,
           a4_results[a4_best_index].threads);
    printf("\nTime = %.6f seconds",a4_results[a4_best_index].time);
    printf("\nRand Index = %.6f",a4_results[a4_best_index].rand_index);
    printf("\n\n========================================================");
    printf("\nA5 ECR + A4 LSH PARAMETER EXPERIMENT");
    printf("\n========================================================");
    printf("\n\nA5 uses eps=%.10f and MinPts=%d selected by ECR.",a5_eps,a5_minpts);
    printf("\nM is fixed at %d; L values are 2,4,6,8,10.",M);

    for(i=0;i<LSH_L_COUNT;i++)
    {
        double total_time[THREAD_COUNT];
        double total_ri[THREAD_COUNT];
        double total_dc[THREAD_COUNT];
        int ti;

        a5_results[i].L=l_values[i];
        a5_results[i].M=M;

        for(ti=0;ti<THREAD_COUNT;ti++)
        {
            total_time[ti]=0.0;
            total_ri[ti]=0.0;
            total_dc[ti]=0.0;
        }

        for(iteration=0;iteration<ITERATIONS;iteration++)
        {
            for(ti=0;ti<THREAD_COUNT;ti++)
            {
                int *predicted=(int *)malloc(sizeof(int)*n);
                int j;
                double start,end,ri;
                long long dc;

                if(predicted==NULL)
                {
                    printf("\nerror: A5 label allocation failed.\n");
                    free(seq_labels); free_memory(); return 1;
                }

                srand(10);
                start=omp_get_wtime();
                dc=dbscan_a4(a5_eps,a5_minpts,threads[ti],l_values[i],M);
                end=omp_get_wtime();

                for(j=0;j<n;j++) predicted[j]=points[j].cluster;
                ri=calculate_rand_index(seq_labels,predicted);

                total_time[ti]+=end-start;
                total_ri[ti]+=ri;
                total_dc[ti]+=(double)dc;
                free(predicted);
            }
        }

        {
            int found=0;
            double best_time_local=1.0e100;
            int best_ti=0;

            for(ti=0;ti<THREAD_COUNT;ti++)
            {
                double avg_time=total_time[ti]/ITERATIONS;
                double avg_ri=total_ri[ti]/ITERATIONS;

                if(avg_ri>=RI_THRESHOLD && avg_time<best_time_local)
                {
                    best_time_local=avg_time;
                    best_ti=ti;
                    found=1;
                }
            }

            if(!found)
            {
                best_ti=0;
                for(ti=1;ti<THREAD_COUNT;ti++)
                {
                    double ri1=total_ri[ti]/ITERATIONS;
                    double ri2=total_ri[best_ti]/ITERATIONS;
                    double time1=total_time[ti]/ITERATIONS;
                    double time2=total_time[best_ti]/ITERATIONS;

                    if(ri1>ri2 || (ri1==ri2 && time1<time2))
                        best_ti=ti;
                }
            }

            a5_results[i].threads=threads[best_ti];
            a5_results[i].time=total_time[best_ti]/ITERATIONS;
            a5_results[i].rand_index=total_ri[best_ti]/ITERATIONS;
            a5_results[i].distance_checks=total_dc[best_ti]/ITERATIONS;
            a5_results[i].speedup=a0_time/a5_results[i].time;
        }

        printf("\nA5 L=%d M=%d best_threads=%d time=%.6f RI=%.6f",
               l_values[i],M,a5_results[i].threads,
               a5_results[i].time,a5_results[i].rand_index);
    }

    a5_best_index=-1;
    for(i=0;i<LSH_L_COUNT;i++)
    {
        if(a5_results[i].rand_index>=RI_THRESHOLD)
        {
            if(a5_best_index==-1 || a5_results[i].time<a5_results[a5_best_index].time)
                a5_best_index=i;
        }
    }

    if(a5_best_index==-1)
    {
        a5_best_index=0;
        for(i=1;i<LSH_L_COUNT;i++)
        {
            if(a5_results[i].rand_index>a5_results[a5_best_index].rand_index ||
              (a5_results[i].rand_index==a5_results[a5_best_index].rand_index &&
               a5_results[i].time<a5_results[a5_best_index].time))
                a5_best_index=i;
        }
    }

    printf("\n\n%-5s%-5s%-9s%-14s%-12s%-12s%-14s", "L","M","Threads","Avg Time","Speedup","Rand Index","Distance Checks");
    printf("\n--------------------------------------------------------------------------------");

    for(i=0;i<LSH_L_COUNT;i++)
    {
        printf("\n%-5d%-5d%-9d%-14.6f%-12.4f%-12.6f%-14.0f",
               a5_results[i].L,a5_results[i].M,a5_results[i].threads,
               a5_results[i].time,a5_results[i].speedup,
               a5_results[i].rand_index,a5_results[i].distance_checks);
    }

    printf("\n\nOptimal A5 configuration:");
    printf("\nECR eps = %.10f, ECR MinPts = %d",a5_eps,a5_minpts);
    printf("\nL = %d, M = %d, threads = %d",
           a5_results[a5_best_index].L,
           a5_results[a5_best_index].M,
           a5_results[a5_best_index].threads);
    printf("\nA5 LSH execution time = %.6f seconds",a5_results[a5_best_index].time);
    printf("\nA5 execution time = LSH + OpenMP + CSR ");
    printf("\nRand Index = %.6f",a5_results[a5_best_index].rand_index);
    printf("\n\n==============================================================");
    printf("\n  A6 HPDBSCAN + CSR EXPERIMENT");
    printf("\n==============================================================");
    printf("\n\nA6 uses the existing selected eps/minpts:");
    printf("\neps = %.10f, MinPts = %d",best_eps,best_minpts);

    {
        method_result r[THREAD_COUNT];

        for(i=0;i<THREAD_COUNT;i++)
        {
            r[i].threads=threads[i];
            r[i].time=0.0;
            r[i].rand_index=0.0;
            r[i].distance_checks=0.0;
        }

        for(iteration=0;iteration<ITERATIONS;iteration++)
        {
            for(i=0;i<THREAD_COUNT;i++)
            {
                int *predicted=(int *)malloc(sizeof(int)*n);
                int j;
                double start,end,ri;
                long long dc;

                if(predicted==NULL)
                {
                    printf("\nerror: A6 label allocation failed.\n");
                    free(seq_labels); free_memory(); return 1;
                }

                start=omp_get_wtime();
                dc=dbscan_a6(best_eps,best_minpts,threads[i]);
                end=omp_get_wtime();

                for(j=0;j<n;j++)
                    predicted[j]=points[j].cluster;

                ri=calculate_rand_index(seq_labels,predicted);

                r[i].time+=end-start;
                r[i].rand_index+=ri;
                r[i].distance_checks+=(double)dc;

                free(predicted);
            }
        }

        for(i=0;i<THREAD_COUNT;i++)
        {
            r[i].time/=ITERATIONS;
            r[i].rand_index/=ITERATIONS;
            r[i].distance_checks/=ITERATIONS;
            r[i].speedup=a0_time/r[i].time;

            printf("\nA6 threads=%d time=%.6f RI=%.6f speedup=%.4f",
                   threads[i],r[i].time,r[i].rand_index,r[i].speedup);

            if(r[i].rand_index>=RI_THRESHOLD)
            {
                if(r[i].time<a6_best.time)
                    a6_best=r[i];
            }
        }

        if(a6_best.time>=1.0e99)
        {
            a6_best=r[0];

            for(i=1;i<THREAD_COUNT;i++)
            {
                if(r[i].rand_index>a6_best.rand_index ||
                  (r[i].rand_index==a6_best.rand_index &&
                   r[i].time<a6_best.time))
                    a6_best=r[i];
            }
        }
    }

    printf("\n\nOptimal A6 configuration:");
    printf("\nThreads = %d",a6_best.threads);
    printf("\nTime = %.6f seconds",a6_best.time);
    printf("\nSpeedup = %.4f",a6_best.speedup);
    printf("\nRand Index = %.6f",a6_best.rand_index);

    {
        FILE *csv=fopen("T_R.csv","w");
        int it,m;
        if(csv!=NULL)
        {
            fprintf(csv,"Iteration,A0 Sequential,A1 OpenMP CSR,A3 Grid OpenMP CSR,A4 LSH OpenMP CSR,A5 ECR LSH,A6 HPDBSCAN\n");
            for(it=0;it<ITERATIONS;it++)
            {
                double t[6]={0,0,0,0,0,0};
                double base_start=omp_get_wtime();
                sequential_dbscan(best_eps,best_minpts);
                t[0]=omp_get_wtime()-base_start;
                {
                    double st=omp_get_wtime(); dbscan_a1(best_eps,best_minpts,a1_best.threads); t[1]=omp_get_wtime()-st;
                }
                {
                    double st=omp_get_wtime(); dbscan_a3(best_eps,best_minpts,a3_best.threads); t[2]=omp_get_wtime()-st;
                }
                {
                    double st=omp_get_wtime(); dbscan_a4(best_eps,best_minpts,a4_results[a4_best_index].threads,a4_results[a4_best_index].L,M); t[3]=omp_get_wtime()-st;
                }
                {
                    double st=omp_get_wtime(); dbscan_a4(a5_eps,a5_minpts,a5_results[a5_best_index].threads,a5_results[a5_best_index].L,M); t[4]=omp_get_wtime()-st;
                }
                {
                    double st=omp_get_wtime(); dbscan_a6(best_eps,best_minpts,a6_best.threads); t[5]=omp_get_wtime()-st;
                }
                for(m=0;m<6;m++)
                    iteration_speedup[m][it]=(m==0)?1.0:(t[m]>0.0?t[0]/t[m]:0.0);
                fprintf(csv,"%d",it+1);
                for(m=0;m<6;m++) fprintf(csv,",%.8f",iteration_speedup[m][it]);
                fprintf(csv,"\n");
            }
            fclose(csv);
        }
        else
            printf("\nWarning: could not create iteration_speedup.csv");
    }

    {
        double global_min_time=a0_time;
        int global_method=0;
        int global_threads=1;
        int global_L=0;
        double global_speedup=1.0;

        if(a1_best.time<global_min_time)
        {
            global_min_time=a1_best.time;
            global_method=1;
            global_threads=a1_best.threads;
            global_speedup=a0_time/a1_best.time;
        }

        if(a3_best.time<global_min_time)
        {
            global_min_time=a3_best.time;
            global_method=3;
            global_threads=a3_best.threads;
            global_speedup=a0_time/a3_best.time;
        }

        if(a4_results[a4_best_index].time<global_min_time)
        {
            global_min_time=a4_results[a4_best_index].time;
            global_method=4;
            global_threads=a4_results[a4_best_index].threads;
            global_L=a4_results[a4_best_index].L;
            global_speedup=a0_time/a4_results[a4_best_index].time;
        }

        if(a5_results[a5_best_index].time<global_min_time)
        {
            global_min_time=a5_results[a5_best_index].time;
            global_method=5;
            global_threads=a5_results[a5_best_index].threads;
            global_L=a5_results[a5_best_index].L;
            global_speedup=a0_time/a5_results[a5_best_index].time;
        }

        if(a6_best.time<global_min_time)
        {
            global_min_time=a6_best.time;
            global_method=6;
            global_threads=a6_best.threads;
            global_L=0;
            global_speedup=a0_time/a6_best.time;
        }
		
        printf("\n\n================================================================================");
        printf("\n                 FINAL A0/A1/A3/A4/A5/A6 COMPARISON");
        printf("\n================================================================================");
        {
            double cm[6],hm[6];
            int *lab=(int *)malloc(sizeof(int)*n);
            int j;
            /* A0 is the reference clustering compared with itself. */
            cm[0]=hm[0]=1.0;
            for(j=1;j<6;j++) cm[j]=hm[j]=0.0;
            if(lab!=NULL)
            {
                dbscan_a1(best_eps,best_minpts,a1_best.threads);
                for(j=0;j<n;j++) lab[j]=points[j].cluster;
                calculate_cm_hm(seq_labels,lab,&cm[1],&hm[1]);
                dbscan_a3(best_eps,best_minpts,a3_best.threads);
                for(j=0;j<n;j++) lab[j]=points[j].cluster;
                calculate_cm_hm(seq_labels,lab,&cm[2],&hm[2]);
                dbscan_a4(best_eps,best_minpts,a4_results[a4_best_index].threads,a4_results[a4_best_index].L,M);
                for(j=0;j<n;j++) lab[j]=points[j].cluster;
                calculate_cm_hm(seq_labels,lab,&cm[3],&hm[3]);
                dbscan_a4(a5_eps,a5_minpts,a5_results[a5_best_index].threads,a5_results[a5_best_index].L,M);
                for(j=0;j<n;j++) lab[j]=points[j].cluster;
                calculate_cm_hm(seq_labels,lab,&cm[4],&hm[4]);
                dbscan_a6(best_eps,best_minpts,a6_best.threads);
                for(j=0;j<n;j++) lab[j]=points[j].cluster;
                calculate_cm_hm(seq_labels,lab,&cm[5],&hm[5]);
                free(lab);
            }
            printf("\n\n%-28s %7s %11s %9s %9s %14s %9s %9s", "Method","Threads","Avg Time","Speedup","RI","ED checks","CM metric","HM metric");
            printf("\n--------------------------------------------------------------------------------------------------------------------------------");
            printf("\n%-28s %7d %11.6f %9.4f %9.6f %14.0f %9.6f %9.6f","A0 Sequential",1,a0_time,1.0,1.0,(double)a0_distances,cm[0],hm[0]);
            printf("\n%-28s %7d %11.6f %9.4f %9.6f %14.0f %9.6f %9.6f","A1 OpenMP + CSR",a1_best.threads,a1_best.time,a0_time/a1_best.time,a1_best.rand_index,a1_best.distance_checks,cm[1],hm[1]);
            printf("\n%-28s %7d %11.6f %9.4f %9.6f %14.0f %9.6f %9.6f","A3 Grid + OpenMP + CSR",a3_best.threads,a3_best.time,a3_best.speedup,a3_best.rand_index,a3_best.distance_checks,cm[2],hm[2]);
            printf("\n%-28s %7d %11.6f %9.4f %9.6f %14.0f %9.6f %9.6f","A4 LSH + OpenMP + CSR",a4_results[a4_best_index].threads,a4_results[a4_best_index].time,a4_results[a4_best_index].speedup,a4_results[a4_best_index].rand_index,a4_results[a4_best_index].distance_checks,cm[3],hm[3]);
            printf("\n%-28s %7d %11.6f %9.4f %9.6f %14.0f %9.6f %9.6f","A5 ECR + A4 LSH",a5_results[a5_best_index].threads,a5_results[a5_best_index].time,a5_results[a5_best_index].speedup,a5_results[a5_best_index].rand_index,a5_results[a5_best_index].distance_checks,cm[4],hm[4]);
            printf("\n%-28s %7d %11.6f %9.4f %9.6f %14.0f %9.6f %9.6f","A6 HPDBSCAN-inspired",a6_best.threads,a6_best.time,a6_best.speedup,a6_best.rand_index,a6_best.distance_checks,cm[5],hm[5]);
        }

        printf("\n\nA4 selected parameters: eps=%.10f MinPts=%d",best_eps,best_minpts);
        printf("\nA5 selected parameters: eps=%.10f MinPts=%d",a5_eps,a5_minpts);
        printf("\nA6 parameters: eps=%.10f MinPts=%d ",best_eps,best_minpts);
        printf("\nRI threshold for an accepted accelerated configuration = %.2f",RI_THRESHOLD);
        printf("\n\n================================================================================");
        printf("\n                       FINAL OPTIMAL METHOD");
        printf("\n================================================================================");

        if(global_method==0)
            printf("\n\nOptimal method : A0 Sequential DBSCAN");
        else if(global_method==1)
            printf("\n\nOptimal method : A1 Redundancy-Reduced OpenMP DBSCAN");
        else if(global_method==3)
            printf("\n\nOptimal method : A3 Hybrid Grid + OpenMP DBSCAN");
        else if(global_method==4)
            printf("\n\nOptimal method : A4 LSH + OpenMP DBSCAN");
        else if(global_method==5)
            printf("\n\nOptimal method : A5 ECR + A4 LSH");
        else
            printf("\n\nOptimal method : A6 HPDBSCAN + OpenMP");

        printf("\nThreads        : %d",global_threads);
        if(global_method==4 || global_method==5)
            printf("\nL              : %d",global_L);
        printf("\nExecution time : %.6f seconds",global_min_time);
        printf("\nSpeedup        : %.4f",global_speedup);
        printf("\nA0 time        : %.6f seconds",a0_time);

        printf("\n\n================================================================================\n");
    }
    free(seq_labels);
    free_memory();
    return 0;
}
