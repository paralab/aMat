// class matvec on gpu used for aMat, allocate according to stream, distinguish independent/dependent elements
// ASSUMPTIONS: all elements in the mesh have the same size of block matrix
#ifndef AMATGPU_H
#define AMATGPU_H

#ifdef USE_GPU
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include "magma_v2.h"
    #include "magma_lapack.h"
#endif

#include <vector>
#include <assert.h>
#include "meshGraph.hpp"

enum class Error {
    SUCCESS, FAILED
};

class aMatGpu {

protected:
    int mi_rank, mi_npes; // rank id, total number of ranks
    unsigned int mui_nStreams; // number of streams
    unsigned int mui_matSize; // matrix size (number of dofs per block)

    unsigned int mui_nElems; // number of independent elements handled by device
    unsigned int mui_nMats; // number of matrices handled by device (i.e. total number of blocks)
    unsigned int mui_numDofsTotal; // number of local dofs (including ghost dofs)

    /* Notes:
    - Matrix ids are determined as follows. From the given local map
    (1) loop over elements, for each element loop over block pointers of the element
    (2) if the block pointer is not nulptr, push back that (non-zero) block to the list of matrices
    - The matrices ids will be in the order of that list 
    - The data structure of element matrices is
    m_epMat = [e1, e2, ..., eN], where ei = std::vector<double*>, some ei can be a nulptr
    for a non-zero block: ei = [p1, p2, ..., pk] where k is the number of blocks for element ei
    pk = [m1, m2, ..., mN] where mi is the ith component of the block k */
    std::vector<unsigned int> m_matId_2_eid;
    std::vector<unsigned int> m_matId_2_blkI;
    std::vector<unsigned int> m_matId_2_blkJ;
    std::vector<unsigned int> m_matId_2_blocksDim;
    std::vector<unsigned int> m_matId_2_blkId;

    unsigned int * m_matId_2_sId;                   // map from matrix id to stream id (i.e. the stream that this matrix is belong to)
    unsigned int * m_matId_2_localStreamMatId;      // map from matrix id to local-to-stream matrix id (i.e. matrix id inside the stream that the matrix is belong to)

    const std::vector<double *> *m_epMat;           // pointer to total elements
    std::vector<unsigned int> m_elemList;           // list of elements handled by gpu
    unsigned int** m_localMap;                      // local map given by user

    magma_queue_t *m_queueStream;                   // queue ID of each stream

    unsigned int *mui_nMatsStream;                  // number of matrices (i.e. blocks) per stream

    std::vector<unsigned int> * m_matIdsPerStream;  // list of matrix ids per stream

    /* address of array of double on device memory holding values of ke matrices, ue/ve vectors */
    double **md_kDevice_s;
    double **md_uDevice_s;
    double **md_vDevice_s;

    /* address of array of double on host memory holding values of ue/ve vectors, must be pinned memory */
    double **md_uHost_s;
    double **md_vHost_s;

    /* address of array of doulbe* on device memory holding address of each ke matrix, ue/ve vector */
    double ***m_kDevAddress_s;
    double ***m_uDevAddress_s;
    double ***m_vDevAddress_s;

    unsigned int mui_nThreads;      // max number of omp threads
    double **md_ueBufs;             // local-to-thread elemental vectors (used in open mp parallel)

public:
    profiler_t gather_time;

public:
    aMatGpu(unsigned int matSize, unsigned int nStreams, MPI_Comm comm);

    ~aMatGpu();

    /* set element matrices */
    Error set_matrix_pointer(const std::vector<double *> *epMat, const std::vector<unsigned int> elemList);

    /* set maps */
    Error set_localMap_pointer(unsigned int **localMap);

    /* set number of local dofs */
    Error set_numDofsTotal(unsigned int numDofsTotal);

    /* compute total non-zero blocks handled by gpu */
    Error compute_nMatrices();

    /* determine number of streams using graph coloring */
    Error compute_nStreams();

    /* setup gpu environment */
    Error setup_environment();

    /* allocate variables */
    Error allocate_variables();

    /* allocate memory on host and device */
    Error allocate_device_host_memory();

    /* loop over elements and transfer element matrix (could contain multiple blocks) to device memory */
    Error transfer_matrices();

    /* loop over elements and copy element vectors to pinned host memory md_uHost */
    Error scatter_u2uHost(double *u);

    /* put elemental vectors back to (local) vector */
    Error gather_vHost2v(double *v);

    /* matrix-vector multiplication */
    Error matvec_v1(); // version 1, loop over streams then inside each loop: (1) transfer, (2) kernel execution, (3) transfer
    Error matvec_v2(); // version 2, (1) loop over streams: transfer, (2) loop over streams: kernel execuation, (3) loop over streams: transfer

protected:

    // allocate local-to-thread vectors ue and ve
    Error allocate_local_thread_ue();

}; // class aMatGpu


// aMatGpu constructor
aMatGpu::aMatGpu(unsigned int matSize, unsigned int nStreams, MPI_Comm comm) {

    // get rank id and total number of ranks
    MPI_Comm_rank(comm, &mi_rank);
    MPI_Comm_size(comm, &mi_npes);

    // set size of matrix, current version works only when all matrices have the same size
    mui_matSize = matSize;
    // set number of streams
    mui_nStreams = nStreams;

    // get number of cpu openMP threads
    mui_nThreads = omp_get_max_threads();
    //printf("number of cpu threads= %d\n", mui_nThreads);

    // allocate local-to-thread element vector
    allocate_local_thread_ue();

    gather_time.clear();
} // aMatGpu constructor


// aMatGpu destructor
aMatGpu::~aMatGpu() {
    // delete device memory holding ke, ue, ve
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_free(md_kDevice_s[sid]);
        magma_free(md_uDevice_s[sid]);
        magma_free(md_vDevice_s[sid]);
    }
    // delete host memory holding pointers to device memory
    delete[] md_kDevice_s;
    delete[] md_uDevice_s;
    delete[] md_vDevice_s;

    // delete host memory holding ue, ve
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_free_pinned(md_uHost_s[sid]);
        magma_free_pinned(md_vHost_s[sid]);
    }
    // delete host memory holding pointers
    delete[] md_uHost_s;
    delete[] md_vHost_s;

    // delete device memory holding address of ke, ue, ve
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_free(m_kDevAddress_s[sid]);
        magma_free(m_uDevAddress_s[sid]);
        magma_free(m_vDevAddress_s[sid]);
    }
    // delete host memory holding pointers to device memory
    delete[] m_kDevAddress_s;
    delete[] m_uDevAddress_s;
    delete[] m_vDevAddress_s;

    // destroy queues, then delete host memory holding queue ids
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_queue_destroy(m_queueStream[sid]);
    }
    delete[] m_queueStream;

    // delete host memory holding number of matrices per stream
    delete[] mui_nMatsStream;

    // delete host memory holding local-to-thread element vectors
    for (unsigned int tid = 0; tid < mui_nThreads; tid++) {
        free(md_ueBufs[tid]);
    }
    free(md_ueBufs);

    delete [] m_matIdsPerStream;

    delete [] m_matId_2_sId;
    delete [] m_matId_2_localStreamMatId;
} // aMatGpu destructor


Error aMatGpu::set_matrix_pointer(const std::vector<double *> *epMat, const std::vector<unsigned int> elemList) {
    m_epMat = epMat;
    m_elemList = elemList;
    mui_nElems = m_elemList.size();
    return Error::SUCCESS;
}

// set maps
Error aMatGpu::set_localMap_pointer(unsigned int **localMap) {
    m_localMap = localMap;
    return Error::SUCCESS;
}

// set number of local dofs
Error aMatGpu::set_numDofsTotal(unsigned int numDofsTotal) {
    mui_numDofsTotal = numDofsTotal;
    return Error::SUCCESS;
}

/* subroutine to determine all non-zero blocks of element matrices. These blocks are the matrices handled by GPU
require: {mui_nElems, m_elemList, m_epMat}
output: {m_matId_2_eid/blkI/blkJ/blocksDim/blkId, mui_nMats} */
Error aMatGpu::compute_nMatrices() {
    for (unsigned int i = 0; i < mui_nElems; i++) {
        const unsigned int eid = m_elemList[i];
        // m_epMat[eid].size() = total number of blocks associated with eid including zero block(s)
        const unsigned int blocks_dim = (unsigned int) sqrt(m_epMat[eid].size());
        assert(blocks_dim * blocks_dim == m_epMat[eid].size());
        for (unsigned int block_i = 0; block_i < blocks_dim; block_i++) {
            for (unsigned int block_j = 0; block_j < blocks_dim; block_j++) {
                const unsigned int block_id = block_i * blocks_dim + block_j;
                if (m_epMat[eid][block_id] != nullptr) {
                    m_matId_2_eid.push_back(eid);
                    m_matId_2_blkI.push_back(block_i);
                    m_matId_2_blkJ.push_back(block_j);
                    m_matId_2_blocksDim.push_back(blocks_dim);
                    m_matId_2_blkId.push_back(block_id);
                }
            }
        }
    }
    // update number of non-zero blocks, this is number of matrices handled by GPU_OVER_CPU
    mui_nMats = m_matId_2_eid.size();

    // for non-stream scatter_u2uHost()
    m_matId_2_sId = new unsigned int[mui_nMats];
    m_matId_2_localStreamMatId = new unsigned int[mui_nMats];

    return Error::SUCCESS;
}

/* subroutine to determine number of streams and list of matrix IDs associated with each stream.
Require: {mui_nMats, mi_rank, mui_numDofsTotal, mui_matSize, m_matId_2_eid/blckI, m_localMap}
Output: {mui_nStreams, m_matIdsPerStream} */
Error aMatGpu::compute_nStreams() {
    meshGraph mGraph(mui_nMats, mi_rank);   // mui_nMats = number of matrices handled by gpu (i.e. non-zero blocks)
    
    /* create a graph from give mesh */
    mGraph.mesh2Graph_xfem(mui_numDofsTotal, mui_nMats, mui_matSize, 
                            m_matId_2_eid.data(), m_matId_2_blkI.data(), m_localMap);
    
    /* coloring the graph */
    mGraph.greedyColoring();
    
    /* get number of colors which is the number of streams */
    mui_nStreams = mGraph.nColors;
    printf("rank %d, number of streams: %d\n", mi_rank, mui_nStreams);

    /* get list of matrices associated with each stream 
    m_matIdsPerStream[sId].size() = number of matrices in stream sId
    m_matIdsPerStream[sId][i] = matrix_Id of matrix i in stream sId */
    m_matIdsPerStream = new std::vector<unsigned int>[mui_nStreams];
    mGraph.computeVerticesPerColor(m_matIdsPerStream);

    /* for using the interface of old code, I compute the number of matrices of each stream here,
    although it can be known from the size() of std::vector */
    mui_nMatsStream = new unsigned int[mui_nStreams];
    for (unsigned int s = 0; s < mui_nStreams; s++) {
        mui_nMatsStream[s] = m_matIdsPerStream[s].size();
        for (unsigned int m = 0; m < m_matIdsPerStream[s].size(); m++) {
            const unsigned int mId = m_matIdsPerStream[s][m];
            m_matId_2_sId[mId] = s;
            m_matId_2_localStreamMatId[mId] = m;
        }
    }
    
    return Error::SUCCESS;
}

/* setup environment for gpu */
Error aMatGpu::setup_environment(){

    // get number of available devices
    int dev_cnt = 0;
    cudaError_t cu_err = cudaGetDeviceCount(&dev_cnt);
    if (cu_err != cudaSuccess || cu_err == cudaErrorNoDevice) {
        std::cout << "Error[] no gpu devices in the node. ::" << std::endl;
        exit(0);
    }
    //if (mi_rank == 0) printf("number of available devices = %d\n", dev_cnt);

    // assign device to my rank
    assert(dev_cnt > 0);
    const int gpu_device_id = mi_rank % dev_cnt;
    magma_setdevice((magma_device_t) gpu_device_id);
    //printf("rank %d, assigned to device %d\n", mi_rank, gpu_device_id);

    // allocate host memory storing queue id of each stream
    m_queueStream = new magma_queue_t[mui_nStreams];
    for (unsigned int s = 0; s < mui_nStreams; s++) {
        m_queueStream[s] = nullptr;
    }

    // create queues (i.e. streams)
    //int deviceId;
    //magma_getdevice(&deviceId);
    //printf("rank %d, device got from magma_getdevice= %d\n", mi_rank, deviceId);
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_queue_create(gpu_device_id, &m_queueStream[sid]);
    }

    return Error::SUCCESS;
}

/* allocate variables of aMatGpu */
Error aMatGpu::allocate_variables() {
    
    // mui_nMatsStream = new unsigned int[mui_nStreams];   /* this was done in compute_nStreams */

    // pointers to device memory holding matrices ke, vectors ue and ve
    md_kDevice_s = new double *[mui_nStreams];
    md_uDevice_s = new double *[mui_nStreams];
    md_vDevice_s = new double *[mui_nStreams];

    /* pointers to device memory holding addresses of each matrice ke, each vector ue and ve */
    m_kDevAddress_s = new double **[mui_nStreams];
    m_uDevAddress_s = new double **[mui_nStreams];
    m_vDevAddress_s = new double **[mui_nStreams];

    /* pointers to host memory (pinned allocated) holding vectors ue and ve */
    md_uHost_s = new double *[mui_nStreams];
    md_vHost_s = new double *[mui_nStreams];

    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        md_kDevice_s[sid] = nullptr;
        md_uDevice_s[sid] = nullptr;
        md_vDevice_s[sid] = nullptr;
        m_kDevAddress_s[sid] = nullptr;
        m_uDevAddress_s[sid] = nullptr;
        m_vDevAddress_s[sid] = nullptr;
    }
    return Error::SUCCESS;
}

// allocate nThreads of double*, inside thread loop will allocate ue so that it is local to thread
Error aMatGpu::allocate_local_thread_ue() {
    md_ueBufs = (double **) malloc(mui_nThreads * sizeof(double *));
    for (unsigned int tid = 0; tid < mui_nThreads; tid++) {
        md_ueBufs[tid] = nullptr;
    }
    return Error::SUCCESS;
} // allocate_local_thread_ue


Error aMatGpu::allocate_device_host_memory() {
    magma_int_t err;

    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        // allocate device memory for each stream, holding values of matrices ke, vectors ue and ve on device
        if (md_kDevice_s[sid] != nullptr) magma_free(md_kDevice_s[sid]);
        err = magma_malloc((void **) &md_kDevice_s[sid],
                           (mui_nMatsStream[sid] * mui_matSize * mui_matSize) * sizeof(double));

        if (md_uDevice_s[sid] != nullptr) magma_free(md_uDevice_s[sid]);
        err = magma_malloc((void **) &md_uDevice_s[sid], (mui_nMatsStream[sid] * mui_matSize) * sizeof(double));

        if (md_vDevice_s[sid] != nullptr) magma_free(md_vDevice_s[sid]);
        err = magma_malloc((void **) &md_vDevice_s[sid], (mui_nMatsStream[sid] * mui_matSize) * sizeof(double));

        // allocate pinned host memory, holding values of vectors ue and ve on host, to be asynchronously transfered to device
        if (md_uHost_s[sid] != nullptr) magma_free_pinned(md_uHost_s[sid]);
        err = magma_malloc_pinned((void **) &md_uHost_s[sid], (mui_nMatsStream[sid] * mui_matSize * sizeof(double)));

        if (md_vHost_s[sid] != nullptr) magma_free_pinned(md_vHost_s[sid]);
        err = magma_malloc_pinned((void **) &md_vHost_s[sid], (mui_nMatsStream[sid] * mui_matSize * sizeof(double)));

        // allocate device memory storing (device-memory) address of each matrix ke, vector ue/ve
        if (m_kDevAddress_s[sid] != nullptr) magma_free(m_kDevAddress_s[sid]);
        err = magma_malloc((void **) &m_kDevAddress_s[sid], mui_nMatsStream[sid] * sizeof(double *));

        if (m_uDevAddress_s[sid] != nullptr) magma_free(m_uDevAddress_s[sid]);
        err = magma_malloc((void **) &m_uDevAddress_s[sid], mui_nMatsStream[sid] * sizeof(double *));

        if (m_vDevAddress_s[sid] != nullptr) magma_free(m_vDevAddress_s[sid]);
        err = magma_malloc((void **) &m_vDevAddress_s[sid], mui_nMatsStream[sid] * sizeof(double *));
    }

    // compute (device-memory) address of each element matrix ke, vector ue and ve
    double ***kAddr_temp = new double **[mui_nStreams];
    double ***uAddr_temp = new double **[mui_nStreams];
    double ***vAddr_temp = new double **[mui_nStreams];
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        kAddr_temp[sid] = new double *[mui_nMatsStream[sid] * sizeof(double *)];
        uAddr_temp[sid] = new double *[mui_nMatsStream[sid] * sizeof(double *)];
        vAddr_temp[sid] = new double *[mui_nMatsStream[sid] * sizeof(double *)];
    }
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        for (unsigned int eid = 0; eid < mui_nMatsStream[sid]; eid++) {
            kAddr_temp[sid][eid] = md_kDevice_s[sid] + eid * (mui_matSize * mui_matSize);
            uAddr_temp[sid][eid] = md_uDevice_s[sid] + eid * mui_matSize;
            vAddr_temp[sid][eid] = md_vDevice_s[sid] + eid * mui_matSize;
        }
    }

    // transfer (device-memory) address from _temp residing on host to device memory
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_setvector(mui_nMatsStream[sid], sizeof(double *), kAddr_temp[sid], 1, m_kDevAddress_s[sid], 1,
                        m_queueStream[sid]);
        magma_setvector(mui_nMatsStream[sid], sizeof(double *), uAddr_temp[sid], 1, m_uDevAddress_s[sid], 1,
                        m_queueStream[sid]);
        magma_setvector(mui_nMatsStream[sid], sizeof(double *), vAddr_temp[sid], 1, m_vDevAddress_s[sid], 1,
                        m_queueStream[sid]);
    }

    // after transfering, delete memory holding device-memory address
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        delete[] kAddr_temp[sid];
        delete[] uAddr_temp[sid];
        delete[] vAddr_temp[sid];
    }
    delete[] kAddr_temp;
    delete[] uAddr_temp;
    delete[] vAddr_temp;

    return Error::SUCCESS;
} // allocate_device_host_memory


// transfer block matrices to device memory
Error aMatGpu::transfer_matrices() {
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        for (unsigned int mid = 0; mid < mui_nMatsStream[sid]; mid++) {
            const unsigned int m = m_matIdsPerStream[sid][mid];
            const unsigned int eid = m_matId_2_eid[m];
            const unsigned int block_id = m_matId_2_blkId[m];
            magma_dsetvector(mui_matSize * mui_matSize, m_epMat[eid][block_id], 1,
                             (md_kDevice_s[sid] + (mid * mui_matSize * mui_matSize)), 1, m_queueStream[sid]);
        }
    }
    /* printf("after transfering from cpu -> gpu, matrices in gpu:\n");
    for (unsigned int matId = 0; matId < mui_nMats; matId++){
       magma_dprint_gpu(mui_matSize * mui_matSize, 1, md_kDevice + (matId * mui_matSize * mui_matSize),
          mui_matSize * mui_matSize, m_queueStream[0]);
    } */

    return Error::SUCCESS;
}// transfer_matrices


// loop over elements, based on the map, extract ue from u and put into md_uHost
Error aMatGpu::scatter_u2uHost(double *u) {
    #pragma omp parallel
    {
        const unsigned int tId = omp_get_thread_num();
        if (md_ueBufs[tId] == nullptr) {
            md_ueBufs[tId] = (double*)malloc(mui_matSize * sizeof(double));
        }
        double* ueLocal = md_ueBufs[tId];
        #pragma omp for
        for (unsigned m = 0; m < mui_nMats; m++) {
            const unsigned int sid = m_matId_2_sId[m];
            const unsigned int mid = m_matId_2_localStreamMatId[m];
            const unsigned int eid = m_matId_2_eid[m];
            const unsigned int block_j = m_matId_2_blkJ[m];
            const unsigned int block_col_offset = block_j * mui_matSize;
            for (unsigned int c = 0; c < mui_matSize; c++) {
                const unsigned int colId = m_localMap[eid][block_col_offset + c];
                ueLocal[c] = u[colId];
                //md_uHost_s[sid][(i * mui_matSize) + c] = u[colId];
            }
            memcpy(&md_uHost_s[sid][mid * mui_matSize], ueLocal, mui_matSize * sizeof(double));
        }
    }
    return Error::SUCCESS;
}


Error aMatGpu::gather_vHost2v(double *v) {
    unsigned int num_finished_streams = 0;
    std::vector<bool> is_stream_done;
    is_stream_done.resize(mui_nStreams, false);
    do {
        for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
            if ((!is_stream_done[sid]) &&
                (cudaStreamQuery(magma_queue_get_cuda_stream(m_queueStream[sid])) == cudaSuccess)) {
                gather_time.start();
                #pragma omp parallel
                {
                    //unsigned int eid, blocks_dim, block_id, block_row_offset, rowId;
                    #pragma omp for
                    for (unsigned int mid = 0; mid < mui_nMatsStream[sid]; mid++) {
                        const unsigned int m = m_matIdsPerStream[sid][mid];
                        const unsigned int eid = m_matId_2_eid[m];
                        const unsigned int block_i = m_matId_2_blkI[m];
                        const unsigned int block_row_offset = block_i * mui_matSize;
                        for (unsigned int r = 0; r < mui_matSize; r++) {
                            const unsigned int rowId = m_localMap[eid][block_row_offset + r];
                            /* no need to omp atomic because the matrices in each stream are completely separate */
                            //#pragma omp atomic
                            v[rowId] += md_vHost_s[sid][(mid * mui_matSize) + r];
                        }
                    }
                }
                num_finished_streams++;
                is_stream_done[sid] = true;
                gather_time.stop();
            }
        }
    } while (num_finished_streams < mui_nStreams);

    return Error::SUCCESS;
} // gather_vHost2v


Error aMatGpu::matvec_v1() {
    double alpha = MAGMA_D_MAKE(1.0, 0.0);
    double beta = MAGMA_D_MAKE(0.0, 0.0);

    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        /* printf("stream %d\n", sid);
        for (unsigned int eid = 0; eid < mui_nElemsStream[sid]; eid++){
           printf("matrix of element %d: ",eid);
           magma_dprint_gpu(mui_matSize * mui_matSize * mui_nMatsStream[sid], 1, md_kDevice_s[sid],
              mui_matSize * mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);
        } */
        // transfer uHost --> uDevice
        magma_dsetvector_async(mui_nMatsStream[sid] * mui_matSize, md_uHost_s[sid], 1, md_uDevice_s[sid], 1,
                               m_queueStream[sid]);

        //printf("u vector in gpu:\n");
        //magma_dprint_gpu(mui_matSize * mui_nMatsStream[sid], 1, md_uDevice_s[sid], mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);

        // batched matrix-vector multiplication
        magmablas_dgemv_batched(MagmaNoTrans, mui_matSize, mui_matSize, alpha, m_kDevAddress_s[sid], mui_matSize,
                                m_uDevAddress_s[sid], 1, beta, m_vDevAddress_s[sid], 1, mui_nMatsStream[sid],
                                m_queueStream[sid]);

        //printf("v vector in gpu:\n");
        //magma_dprint_gpu(mui_matSize * mui_nMatsStream[sid], 1, md_vDevice_s[sid], mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);

        // transfer vDevice --> vHost
        magma_dgetvector_async(mui_nMatsStream[sid] * mui_matSize, md_vDevice_s[sid], 1, md_vHost_s[sid], 1,
                               m_queueStream[sid]);

        //printf("after batched matvec, v vectors in gpu:\n");
        //magma_dprint_gpu(mui_matSize * mui_nMatsStream[sid], 1, md_vDevice_s[sid], mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);
    }
    return Error::SUCCESS;
} // matvec_v1


Error aMatGpu::matvec_v2() {

    // transfer uHost --> uDevice
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_dsetvector_async(mui_nMatsStream[sid] * mui_matSize, md_uHost_s[sid], 1, md_uDevice_s[sid], 1,
                               m_queueStream[sid]);
    }

    // batched matrix-vector multiplication
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magmablas_dgemv_batched(MagmaNoTrans, mui_matSize, mui_matSize, 1.0, m_kDevAddress_s[sid], mui_matSize,
                                m_uDevAddress_s[sid], 1, 0.0, m_vDevAddress_s[sid], 1, mui_nMatsStream[sid],
                                m_queueStream[sid]);
    }

    // transfer vDevice --> vHost
    for (unsigned int sid = 0; sid < mui_nStreams; sid++) {
        magma_dgetvector_async(mui_nMatsStream[sid] * mui_matSize, md_vDevice_s[sid], 1, md_vHost_s[sid], 1,
                               m_queueStream[sid]);
    }

    return Error::SUCCESS;
} // matvec_v2

#endif //AMATGPU_H