# Aug 25, 2021

In the meeting with Keith and Hari on Aug 17, we agreed to merge aMat with aMat_for_paper so that aMat is updated with all features of aMat_for_paper including GPU. After this merge, the merge with aMat_dev (no Eigen, no block version) will be proceeded.

## include/aMat.hpp
Add the following:  
```c
#include "ke_matrix.hpp"
#include "lapack_extern.hpp"
```
```c
#ifdef USE_GPU
    #include "aMatGpu.hpp"
    #include <cuda.h>
    #include "magma_v2.h"
    #include "magma_lapack.h"
#endif
```
```c
MATFREE_TYPE m_freeType; // hybrid or free for matrix-free method
```
```c
void set_element_matrix_function(void (*eMat)(LI, DT*, DT*)){
        m_eleMatFunc = eMat;
}
```
```c
/**@brief set matrix-free type */
Error set_matfree_type(MATFREE_TYPE type){
    m_freeType = type;
    return Error::SUCCESS;
}

/**@brief set number of streams if GPU is used */
#ifdef USE_GPU
Error set_num_streams(LI nStreams){
    return static_cast<Derived*>(this)->set_num_streams(nStreams);
}
#endif

/**@brief v = aMat_matrix * u, using quasi matrix free */
Error matvec(DT* v, const DT* u, bool isGhosted) {
    return static_cast<Derived*>(this)->matvec(v, u, isGhosted);
}

/**@brief v = aMat_matrix * u, using Petsc */
Error matmult(Vec v, Vec u) {
    return static_cast<Derived*>(this)->matmult(v, u);
}
```
## include/aMatBased.hpp  
```c
using ParentType::m_freeType;
```
```c
/*@brief v = aMat_matrix * u, usef for profiling */
Error matmult(Vec v, Vec u);
```
```c
template<typename DT, typename GI, typename LI>
Error aMatBased<DT, GI, LI>::matmult(Vec v, Vec u){
    MatMult(m_pMat, u, v);
    return Error::SUCCESS;
}
```
## include/aMatFree.hpp  
1. Add the following  
```c
using ParentType::m_freeType; // hybrid or free or gpu
using ParentType::m_eleMatFunc;
```
2. Delete #ifdef HYBRID_PARALLEL made for m_uiNumThreads, m_veBufs, m_ueBufs (i.e. they are now available even for case of not using hybrid paralel )  
add the following
```c
DT** m_xeBufs;
DT*** m_keBufs;
DT* m_xe;
DT** m_ke;
```
```c
// constants used in complete-matrix-free approach
const unsigned int MAX_DOFS_PER_NODE = 3;
const unsigned int MAX_DIMS = 3;
const unsigned int MAX_NODES_PER_BLOCK = 20;
const unsigned int MAX_BLOCKS_PER_ELEMENT = 1;
const unsigned int MAX_DOFS_PER_BLOCK = MAX_DOFS_PER_NODE * MAX_NODES_PER_BLOCK;

#ifdef VECTORIZED_OPENMP_ALIGNED
    const unsigned int MAX_PADS = ALIGNMENT/sizeof(DT) - MAX_DOFS_PER_BLOCK % (ALIGNMENT/sizeof(DT));
#else
    const unsigned int MAX_PADS = 0;
#endif

//2021.01.06
#ifdef USE_GPU
    aMatGpu * m_aMatGpu; // for computing all elements (option 3) or only independent elements (option 4)
    aMatGpu * m_aMatGpu_dep; // for computing dependent elements (option 5)
    LI m_uiNumStreams = 0; // todo: user inputs number of streams
#endif
```
3. The following function is now has definition and is used for gpu case (no logner "not applicable")  
```c
/**@brief initialize for gpu options */
Error finalize_begin();
```
4. Move the definition of following function outside of class body
```c
/**@brief compute trace of matrix used in penalty method */
Error finalize_end();
```
5. Make both matvec_ghosted_OMP() and matvec_ghosted_noOMP available (no longer in #ifdef HYBRID_PARALLEL)
6. Add the following
```c
Error matvec_ghosted_OMP_free(DT* v, DT* u); // matrix-free version
Error matvec_ghosted_noOMP_free(DT* v, DT* u);// matrix-free version
```
```c
#ifdef USE_GPU
    /**@brief initialize gpu */
    Error init_gpu();
    /**@brief finalize gpu */
    Error finalize_gpu();
    /**@brief matrix-vector multiplication using gpu */
    Error matvec_ghosted_gpuPure(DT* v, DT*u); // all elements on gpu
    Error matvec_ghosted_gpuOverCpu(DT *v, DT *u); // independent elements on gpu overlap with dependent elements on cpu
    Error matvec_ghosted_gpuOverGpu(DT *v, DT *u); // independent elements on gpu overlap with dependent elements on gpu
#endif
```
## include/enums.hpp  
1. Add the following  
```c
// UNDEFINED implicitly refer to matrix-based approach
enum class MATFREE_TYPE {UNDEFINED = 0, QUASI_FREE = 1, FREE = 2, GPU_PURE = 3, GPU_OVER_CPU = 4, GPU_OVER_GPU = 5};
```
## replace all files in examples/include and examples/src by all files in examples/include and examples/src of aMat_for_paper, respectively
