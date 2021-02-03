// class matvec on gpu used for aMat, allocate according to stream
#ifndef AMATGPU_S_H
#define AMATGPU_S_H

#include <cuda.h>
#include <cuda_runtime.h>
#include "magma_v2.h"
#include "magma_lapack.h"

#include <vector>
#include <assert.h>

enum class Error {SUCCESS, FAILED};

class aMatGpu_s {

   protected:
   unsigned int mui_nStreams; // number of streams
   unsigned int mui_matSize; // matrix size (number of dofs per block)

   // number of elements per stream, currently we distribute elements uniformly to streams
   unsigned int* mui_nElemsStream;

   // offset of number of elements per stream
   unsigned int* mui_nElemsStreamOffset;

   // number of matrices/blocks per stream
   unsigned int* mui_nMatsStream;

   // queue ID of each stream
   magma_queue_t * m_queueStream;

   // pointer to elemental matrices
   std::vector<double*>* m_epMat;

   // pointer to the map
   unsigned int* * m_map;

   int mi_device; // device ID

   // address of array of double on device memory holding values of ke matrices, ue/ve vectors
   double * * md_kDevice_s;
   double * * md_uDevice_s;
   double * * md_vDevice_s;

   // address of array of double on host memory holding values of ue/ve vectors, must be pinned memory
   double * * md_uHost_s;
   double * * md_vHost_s;

   // address of array of doulbe* on device memory holding address of each ke matrix, ue/ve vector
   double* * * m_kDevAddress_s;
   double* * * m_uDevAddress_s;
   double* * * m_vDevAddress_s;

   unsigned int m_uiNumThreads; // max number of omp threads
   double** m_ueBufs;               // local-to-thread elemental vectors (used in open mp parallel)

   // ========== methods ==========
   public:
   aMatGpu_s(unsigned int nElems, unsigned int matSize, std::vector<double*>* epMat, unsigned int* * map, 
            unsigned int nStreams, MPI_Comm comm);
   ~aMatGpu_s();

   // loop over elements and transfer element matrix (could contain multiple blocks) to device memory
   Error transfer_matrices();
   
   // loop over elements and copy element vectors to pinned host memory md_uHost
   Error scatter_u2uHost(double* u);
   
   // put 
   Error gather_vHost2v(double* v);

   // matrix-vector multiplication
   Error matvec_v1(); // version 1
   Error matvec_v2(); // version 2

   // synchronize host with operations in stream
   Error synchronize_stream();
   
   protected:
   // compute number of matrices for each stream
   Error compute_n_matrices_stream(unsigned int nTotalMatrices, unsigned int nStreams, unsigned int* nMatsStream);

   // compute number of elements for each stream
   Error compute_n_elements_stream();

   // allocate local-to-thread vectors ue and ve
   Error allocate_ue_ve();

}; // class aMatGpu_s


// aMatGpu constructor
aMatGpu_s::aMatGpu_s(unsigned int nElems, unsigned int matSize, std::vector<double*>* epMat, unsigned int* * map, 
                     unsigned int nStreams, MPI_Comm comm) {
   
   // get rank id
   int rank, npes;
   MPI_Comm_rank(comm,&rank);
   MPI_Comm_size(comm,&npes);

   /* if (rank == 0)
        setenv( "CUDA_VISIBLE_DEVICES", "0", 1 );
   else
        setenv( "CUDA_VISIBLE_DEVICES", "1", 1 ); */

   // get number of devices available
   int dev_cnt = 0;
   cudaError_t cu_err = cudaGetDeviceCount( &dev_cnt );

   if(cu_err != cudaSuccess || cu_err == cudaErrorNoDevice) {
      std::cout<<"Error[] no gpu devices in the node. ::"<<std::endl;
      exit(0);
   }

   // assign equally devices to rank
   assert(dev_cnt>0);
   const int gpu_device_id =  rank % dev_cnt;

   magma_setdevice((magma_device_t)gpu_device_id);

   mui_matSize = matSize;
   m_epMat = epMat;
   m_map = map;
   mui_nStreams = nStreams;
   
   // allocate host memory holding number of elements per stream
   mui_nElemsStream = new unsigned int [mui_nStreams];

   // compute number of elements per stream
   // todo: distribute elements to streams based on number of blocks
   compute_n_matrices_stream(nElems, nStreams, mui_nElemsStream);
   /* for (unsigned int s = 0; s < mui_nStreams; s++){
      printf("nElemsStream[%d]= %d\n", s, mui_nElemsStream[s]);
   } */

   // compute offset of number of matrices per stream
   mui_nElemsStreamOffset = new unsigned int [mui_nStreams];
   mui_nElemsStreamOffset[0] = 0;
   for (unsigned int sid = 1; sid < mui_nStreams; sid++){
      mui_nElemsStreamOffset[sid] = mui_nElemsStreamOffset[sid - 1] + mui_nElemsStream[sid - 1];
   }

   // compute number of matrices (ie. non-zero blocks) per stream
   mui_nMatsStream = new unsigned int [mui_nStreams];
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      unsigned int nnzBlocks = 0;
      for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
         const unsigned int eid = mui_nElemsStreamOffset[sid] + i;
         const unsigned int blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
         for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
            for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
               const unsigned int block_id = block_i * blocks_dim + block_j;
               if (m_epMat[eid][block_id] != nullptr){
                  nnzBlocks += 1;
               }
            }
         }
      }
      mui_nMatsStream[sid] = nnzBlocks;
   }

   magma_int_t err;

   // allocate device memory, storing values of matrices ke, vectors ue and ve on device
   md_kDevice_s = new double* [mui_nStreams];
   md_uDevice_s = new double* [mui_nStreams];
   md_vDevice_s = new double* [mui_nStreams];
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      err = magma_malloc((void**)&md_kDevice_s[sid], (mui_nMatsStream[sid] * mui_matSize * mui_matSize) * sizeof(double));
      err = magma_malloc((void**)&md_uDevice_s[sid], (mui_nMatsStream[sid] * mui_matSize) * sizeof(double));
      err = magma_malloc((void**)&md_vDevice_s[sid], (mui_nMatsStream[sid] * mui_matSize) * sizeof(double));
   }
   
   // allocate pinned host memory, storing values of vectors ue and ve on host, to be asynchronously transfered to device
   md_uHost_s = new double* [mui_nStreams];
   md_vHost_s = new double* [mui_nStreams];
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      err = magma_malloc_pinned((void**)&md_uHost_s[sid], (mui_nMatsStream[sid] * mui_matSize * sizeof(double)));
      err = magma_malloc_pinned((void**)&md_vHost_s[sid], (mui_nMatsStream[sid] * mui_matSize * sizeof(double)));
   }
   
   // allocate device memory storing (device-memory) address of each matrix ke, vector ue/ve
   m_kDevAddress_s = new double** [mui_nStreams];
   m_uDevAddress_s = new double** [mui_nStreams];
   m_vDevAddress_s = new double** [mui_nStreams];
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      err = magma_malloc((void**)&m_kDevAddress_s[sid], mui_nMatsStream[sid] * sizeof(double*));
      err = magma_malloc((void**)&m_uDevAddress_s[sid], mui_nMatsStream[sid] * sizeof(double*));
      err = magma_malloc((void**)&m_vDevAddress_s[sid], mui_nMatsStream[sid] * sizeof(double*));
   }
   
   // compute (device-memory) address of each element matrix ke, vector ue and ve
   double * * * kAddr_temp = new double** [mui_nStreams];
   double * * * uAddr_temp = new double** [mui_nStreams];
   double * * * vAddr_temp = new double** [mui_nStreams];
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      kAddr_temp[sid] = new double* [mui_nMatsStream[sid] * sizeof(double*)];
      uAddr_temp[sid] = new double* [mui_nMatsStream[sid] * sizeof(double*)];
      vAddr_temp[sid] = new double* [mui_nMatsStream[sid] * sizeof(double*)];
   }
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      for (unsigned int eid = 0; eid < mui_nMatsStream[sid]; eid++){
         kAddr_temp[sid][eid] = md_kDevice_s[sid] + eid * (mui_matSize * mui_matSize);
         uAddr_temp[sid][eid] = md_uDevice_s[sid] + eid * mui_matSize;
         vAddr_temp[sid][eid] = md_vDevice_s[sid] + eid * mui_matSize;
      }
   }

   // allocate host memory storing queue id of each stream
   m_queueStream = new magma_queue_t [mui_nStreams];
   for (unsigned int s = 0; s < mui_nStreams; s++) m_queueStream[s] = nullptr;
   
   // create queues (i.e. streams)
   magma_getdevice(&mi_device);
   //printf("rank %d, device_id= %d, mi_device= %d\n", rank, gpu_device_id, mi_device);
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_queue_create(mi_device, &m_queueStream[sid]);
   }

   // transfer (device-memory) address from _temp residing on host to device memory
   for (unsigned int sid = 0; sid < nStreams; sid++){
      magma_setvector(mui_nMatsStream[sid], sizeof(double*), kAddr_temp[sid], 1, m_kDevAddress_s[sid], 1, m_queueStream[sid]);
      magma_setvector(mui_nMatsStream[sid], sizeof(double*), uAddr_temp[sid], 1, m_uDevAddress_s[sid], 1, m_queueStream[sid]);
      magma_setvector(mui_nMatsStream[sid], sizeof(double*), vAddr_temp[sid], 1, m_vDevAddress_s[sid], 1, m_queueStream[sid]);
   }

   allocate_ue_ve();

   // after transfering, delete memory holding device-memory address
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      delete [] kAddr_temp[sid];
      delete [] uAddr_temp[sid];
      delete [] vAddr_temp[sid];
   }
   delete [] kAddr_temp;
   delete [] uAddr_temp;
   delete [] vAddr_temp;

} // aMatGpu_s constructor


// aMatGpu_s destructor
aMatGpu_s::~aMatGpu_s() {
   // delete device memory holding ke, ue, ve
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_free(md_kDevice_s[sid]);
      magma_free(md_uDevice_s[sid]);
      magma_free(md_vDevice_s[sid]);
   }
   // delete host memory holding pointers to device memory
   delete [] md_kDevice_s;
   delete [] md_uDevice_s;
   delete [] md_vDevice_s;

   // delete host memory holding ue, ve
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_free_pinned(md_uHost_s[sid]);
      magma_free_pinned(md_vHost_s[sid]);
   }
   // delete host memory holding pointers
   delete [] md_uHost_s;
   delete [] md_vHost_s;

   // delete device memory holding address of ke, ue, ve
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_free(m_kDevAddress_s[sid]);
      magma_free(m_uDevAddress_s[sid]);
      magma_free(m_vDevAddress_s[sid]);
   }
   // delete host memory holding pointers to device memory
   delete [] m_kDevAddress_s;
   delete [] m_uDevAddress_s;
   delete [] m_vDevAddress_s;

   // destroy queues, then delete host memory holding queue ids
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_queue_destroy(m_queueStream[sid]);
   }
   delete [] m_queueStream;

   // delete host memory holding number of matrices per stream
   delete [] mui_nMatsStream;

   // delete host memory holding number of elements per stream
   delete [] mui_nElemsStreamOffset;

   // delete host memory holding offset of elements per stream
   delete [] mui_nElemsStream;

   // delete host memory holding local-to-thread element vectors
   for (unsigned int tid = 0; tid < m_uiNumThreads; tid++) {
      free(m_ueBufs[tid]);
   }
   free(m_ueBufs);

} // aMatGpu_s destructor


Error aMatGpu_s::allocate_ue_ve() {
   m_uiNumThreads = omp_get_max_threads(); // max of omp threads
   m_ueBufs = (double**)malloc(m_uiNumThreads * sizeof(double*));
   for (unsigned int tid = 0; tid < m_uiNumThreads; tid++) {
      m_ueBufs[tid] = nullptr;
   }
   return Error::SUCCESS;
} // allocate_ue_ve


// compute number of matrices per stream
Error aMatGpu_s::compute_n_matrices_stream(unsigned int nTotalMatrices, unsigned int nStreams, unsigned int* nMatsStream) {
   const unsigned int residual = nTotalMatrices % nStreams;
   if (residual == 0){
      for (unsigned int sid = 0; sid < nStreams; sid++){
            nMatsStream[sid] = nTotalMatrices / nStreams;
      }
   } else {
      for (unsigned int sid = 0; sid < residual; sid++){
            nMatsStream[sid] = ((nTotalMatrices - residual) / nStreams) + 1;
      }
      for (unsigned int sid = residual; sid < nStreams; sid++){
            nMatsStream[sid] = ((nTotalMatrices - residual) / nStreams);
      }
   }
   return Error::SUCCESS;
} // compute_n_matrices_stream


// compute number of elements per stream
Error aMatGpu_s::compute_n_elements_stream(){
   
}


// transfer block matrices to device memory
Error aMatGpu_s::transfer_matrices(){
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      unsigned int matId = 0;
      for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
         // global element id
         const unsigned int eid = mui_nElemsStreamOffset[sid] + i;
         unsigned int blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
         for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
            for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
               unsigned int block_id = block_i * blocks_dim + block_j;
               if (m_epMat[eid][block_id] != nullptr){
                  // transfer block_id to the position of matId on device memory md_kDevice
                  magma_dsetvector(mui_matSize * mui_matSize, m_epMat[eid][block_id], 1, 
                           (md_kDevice_s[sid] + (matId * mui_matSize * mui_matSize)), 1, m_queueStream[sid]);
                  // move matId to the next one
                  matId += 1;
               }
            }
         }
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
/* Error aMatGpu_s::scatter_u2uHost(double* u){
   unsigned int matId, colId, blocks_dim, block_id, block_col_offset, eid;
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      matId = 0;
      for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
         eid = mui_nElemsStreamOffset[sid] + i;
         blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
         for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
            for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
               block_col_offset = block_j * mui_matSize;
               block_id = block_i * blocks_dim + block_j;
               if (m_epMat[eid][block_id] != nullptr){
                  for (unsigned int c = 0; c < mui_matSize; c++){
                     colId = m_map[eid][block_col_offset + c];
                     md_uHost_s[sid][(matId * mui_matSize) + c] = u[colId];
                  }
                  // move matId to the next one
                  matId += 1;
               }
            }
         }
      }
   }
   return Error::SUCCESS;
} */ // scatter_u2uHost


// loop over elements, based on the map, extract ue from u and put into md_uHost
Error aMatGpu_s::scatter_u2uHost(double* u){
   //unsigned int matId, colId, blocks_dim, block_id, block_col_offset, eid;
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      #pragma omp parallel
      {
         //unsigned int eid, blocks_dim, block_col_offset, block_id, colId;

         // get thread id
         /* const unsigned int tId = omp_get_thread_num();
         if (m_ueBufs[tId] == nullptr) {
            m_ueBufs[tId] = (double*)malloc(mui_matSize * sizeof(double));
         }
         double* ueLocal = m_ueBufs[tId]; */

         //unsigned int matId = 0;

         #pragma omp for
         for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
            const unsigned int eid = mui_nElemsStreamOffset[sid] + i;
            const unsigned int blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
            for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
               for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
                  const unsigned int block_col_offset = block_j * mui_matSize;
                  const unsigned int block_id = block_i * blocks_dim + block_j;
                  if (m_epMat[eid][block_id] != nullptr){
                     for (unsigned int c = 0; c < mui_matSize; c++){
                        const unsigned int colId = m_map[eid][block_col_offset + c];
                        //ueLocal[c] = u[colId];
                        md_uHost_s[sid][(i * mui_matSize) + c] = u[colId];
                     }
                     // copy entire ue to uHost_s
                     // todo: currently, only 1 block per element thus eid is correct --> fix later
                     //memcpy(&md_uHost_s[sid][(i * mui_matSize)], ueLocal, mui_matSize * sizeof(double));
                     // move matId to the next one
                     //matId += 1;
                  }
               }
            }
         }
      }
   }
   return Error::SUCCESS;
} // scatter_u2uHost
   

// loop over elements, based on the map, "scatter" ve from md_vHost to v
/* Error aMatGpu_s::gather_vHost2v(double* v){
   //unsigned int matId, rowId, blocks_dim, block_id, block_row_offset, eid;
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      // synchronize stream to make sure all operations in stream are done
      magma_queue_sync(m_queueStream[sid]);
      // initialize matrix index before looping on matrices (blocks) of element
      unsigned int matId = 0;
      for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
         const unsigned int eid = mui_nElemsStreamOffset[sid] + i;
         const unsigned int blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
         for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
            const unsigned int block_row_offset = block_i * mui_matSize;
            for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
               const unsigned int block_id = block_i * blocks_dim + block_j;
               if (m_epMat[eid][block_id] != nullptr){
                  for (unsigned int r = 0; r < mui_matSize; r++){
                     const unsigned int rowId = m_map[eid][block_row_offset + r];
                     v[rowId] += md_vHost_s[sid][(matId * mui_matSize) + r];
                  }
                  // move matId to the next one
                  matId += 1;
               }
            }
         }
      }
   }
   return Error::SUCCESS;
}  */// gather_vHost2v

Error aMatGpu_s::gather_vHost2v(double* v){
   //unsigned int matId, rowId, blocks_dim, block_id, block_row_offset, eid;
   unsigned int num_finished_streams=0;
   std::vector<bool> is_stream_done;
   is_stream_done.resize(mui_nStreams,false);
   do
   {
      for (unsigned int sid = 0; sid < mui_nStreams; sid++){
         if ( (!is_stream_done[sid]) && (cudaStreamQuery(magma_queue_get_cuda_stream(m_queueStream[sid])) == cudaSuccess)){
            #pragma omp parallel
            {
               //unsigned int eid, blocks_dim, block_id, block_row_offset, rowId;
               #pragma omp for
               for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
                  const unsigned int eid = mui_nElemsStreamOffset[sid] + i;
                  const unsigned int blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
                  for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
                     for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
                        const unsigned int block_row_offset = block_i * mui_matSize;
                        const unsigned int block_id = block_i * blocks_dim + block_j;
                        if (m_epMat[eid][block_id] != nullptr){
                           for (unsigned int r = 0; r < mui_matSize; r++){
                              const unsigned int rowId = m_map[eid][block_row_offset + r];
                              #pragma omp atomic
                              v[rowId] += md_vHost_s[sid][(i * mui_matSize) + r];
                           }
                        }
                     }
                  }
               }
            }
            num_finished_streams++;
            is_stream_done[sid]=true;
         } 
      }
   } while(num_finished_streams < mui_nStreams);

   /* for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_queue_sync(m_queueStream[sid]);
      #pragma omp parallel
      {
         //unsigned int eid, blocks_dim, block_id, block_row_offset, rowId;
         #pragma omp for
         for (unsigned int i = 0; i < mui_nElemsStream[sid]; i++){
            const unsigned int eid = mui_nElemsStreamOffset[sid] + i;
            const unsigned int blocks_dim = (unsigned int)sqrt(m_epMat[eid].size());
            for (unsigned int block_i = 0; block_i < blocks_dim; block_i++){
               for (unsigned int block_j = 0; block_j < blocks_dim; block_j++){
                  const unsigned int block_row_offset = block_i * mui_matSize;
                  const unsigned int block_id = block_i * blocks_dim + block_j;
                  if (m_epMat[eid][block_id] != nullptr){
                     for (unsigned int r = 0; r < mui_matSize; r++){
                        const unsigned int rowId = m_map[eid][block_row_offset + r];
                        #pragma omp atomic
                        v[rowId] += md_vHost_s[sid][(i * mui_matSize) + r];
                     }
                  }
               }
            }
         }
      }
   } */

   return Error::SUCCESS;
} // gather_vHost2v


Error aMatGpu_s::matvec_v1(){
   double alpha = MAGMA_D_MAKE(1.0, 0.0);
   double beta = MAGMA_D_MAKE(0.0, 0.0);

   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      /* printf("stream %d\n", sid);
      for (unsigned int eid = 0; eid < mui_nElemsStream[sid]; eid++){
         printf("matrix of element %d: ",eid);
         magma_dprint_gpu(mui_matSize * mui_matSize * mui_nMatsStream[sid], 1, md_kDevice_s[sid], 
            mui_matSize * mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);
      } */
      // transfer uHost --> GPU
      magma_dsetvector_async(mui_nMatsStream[sid] * mui_matSize, md_uHost_s[sid], 1, md_uDevice_s[sid], 1, m_queueStream[sid]);

      //printf("u vector in gpu:\n");
      //magma_dprint_gpu(mui_matSize * mui_nMatsStream[sid], 1, md_uDevice_s[sid], mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);
      
      // batched matrix-vector multiplication
      magmablas_dgemv_batched(MagmaNoTrans, mui_matSize, mui_matSize, alpha, m_kDevAddress_s[sid], mui_matSize,
                  m_uDevAddress_s[sid], 1, beta, m_vDevAddress_s[sid], 1, mui_nMatsStream[sid], m_queueStream[sid]);
      
      //printf("v vector in gpu:\n");
      //magma_dprint_gpu(mui_matSize * mui_nMatsStream[sid], 1, md_vDevice_s[sid], mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);

      // transfer GPU --> vHost
      magma_dgetvector_async(mui_nMatsStream[sid] * mui_matSize, md_vDevice_s[sid], 1, md_vHost_s[sid], 1, m_queueStream[sid]);

      //printf("after batched matvec, v vectors in gpu:\n");
      //magma_dprint_gpu(mui_matSize * mui_nMatsStream[sid], 1, md_vDevice_s[sid], mui_matSize * mui_nMatsStream[sid], m_queueStream[sid]);
   }
   return Error::SUCCESS;
} // matvec_v1


Error aMatGpu_s::matvec_v2(){
   
   // transfer uHost --> GPU
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_dsetvector_async(mui_nMatsStream[sid] * mui_matSize, md_uHost_s[sid], 1, md_uDevice_s[sid], 1, m_queueStream[sid]);
   }
   
   // batched matrix-vector multiplication
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magmablas_dgemv_batched(MagmaNoTrans, mui_matSize, mui_matSize, 1.0, m_kDevAddress_s[sid], mui_matSize,
                  m_uDevAddress_s[sid], 1, 0.0, m_vDevAddress_s[sid], 1, mui_nMatsStream[sid], m_queueStream[sid]);
   }   

   // transfer GPU --> vHost
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_dgetvector_async(mui_nMatsStream[sid] * mui_matSize, md_vDevice_s[sid], 1, md_vHost_s[sid], 1, m_queueStream[sid]);
   }

   return Error::SUCCESS;
} // matvec_v2


// synchronize host with operations in stream
Error aMatGpu_s::synchronize_stream(){
   for (unsigned int sid = 0; sid < mui_nStreams; sid++){
      magma_queue_sync(m_queueStream[sid]);
   }
   return Error::SUCCESS;
}

#endif //AMATGPU_S_H