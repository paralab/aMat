/**
 * @file maps.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Han Duc Tran     hantran@cs.utah.edu
 *
 * @brief A class to manage maps used in matrix-based and/or matrix-free methods for adaptive finite elements. 
 * 
 * @version 0.1
 * @date 2020-06-12
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_MAPS_H
#define ADAPTIVEMATRIX_MAPS_H

#include <Eigen/Dense>

#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include "asyncExchangeCtx.hpp"
#include "enums.hpp"

namespace par {

//==============================================================================================================
// Class Maps
template <typename DT, typename GI, typename LI>
class Maps {

public:
    typedef DT DTType;
    typedef GI GIType;
    typedef LI LIType;

protected:
    MPI_Comm m_comm; // communicator
    unsigned int m_uiRank; // my rank id
    unsigned int m_uiSize; // total number of ranks

    LI m_uiNumDofs; // number of dofs owned by my rank
    LI m_uiNumDofsTotal; // total number of dofs inclulding ghost dofs

    GI m_ulGlobalDofStart; // start of global dof ID owned by my rank
    GI m_ulGlobalDofEnd; // end of global dof ID owned by my rank (# owned dofs = m_ulGlobalDofEnd - m_ulGlobalDofStart + 1)

    GI m_ulNumDofsGlobal; // number of DoFs owned by all ranks

    LI m_uiNumElems; // number of elements owned by rank

    GI** m_ulpMap; // element-to-global map, i.e. m_ulpMap[eid][element_dof_id]  = global dof id

    const LI* m_uiDofsPerElem; // number of dofs per element

    unsigned int** m_uipBdrMap; // element-to-constrain_flag map, i.e. 0 = free dof, 1 = constrained dof
    DT** m_dtPresValMap; // element-to-prescribed_value map
    std::vector<GI> ownedConstrainedDofs; // list of constrained DOFs owned by my rank
    std::vector<DT> ownedPrescribedValues; // list of values prescribed at constrained DOFs owned by my rank
    std::vector<GI> ownedFreeDofs; // list of free DOFs owned by my rank

    LI n_owned_constraints; // number of owned constraints

    LI** m_uipLocalMap; // element-to-local map, i.e. m_uipLocalMap[eid][element_dof_id]  = local dof id

    GI* m_ulpLocal2Global; // map from local dof id to global dof id

    std::vector<LI> m_uivLocalDofCounts; // number of DoFs owned by each rank, NOT include ghost DoFs
    std::vector<LI> m_uivLocalElementCounts; // number of elements owned by each rank
    std::vector<GI> m_ulvLocalDofScan; // exclusive scan of (local) number of DoFs
    std::vector<GI> m_ulvLocalElementScan; // exclusive scan of (local) number of elements

    LI m_uiNumPreGhostDofs; // number of ghost DoFs owned by "pre" processes (whose ranks are smaller than m_uiRank)
    LI m_uiNumPostGhostDofs; // total number of ghost DoFs owned by "post" processes (whose ranks are larger than m_uiRank)

    std::vector<LI> m_uivSendDofCounts; // number of DoFs sent to each process (size = m_uiSize)
    std::vector<LI> m_uivSendDofOffset; // offsets (i.e. exclusive scan) of m_uiSendNodeCounts

    std::vector<LI> m_uivSendDofIds; // local DoF IDs to be sent (size = total number of nodes to be sent */

    std::vector<unsigned int> m_uivSendRankIds; // process IDs that I send data to

    std::vector<LI> m_uivRecvDofCounts; // number of DoFs to be received from each process (size = m_uiSize)
    std::vector<LI> m_uivRecvDofOffset; // offsets (i.e. exclusive scan) of m_uiRecvNodeCounts

    std::vector<unsigned int> m_uivRecvRankIds; // process IDs that I receive data from

    LI m_uiDofPreGhostBegin; // local dof-ID starting of pre-ghost nodes, always = 0
    LI m_uiDofPreGhostEnd; // local dof-ID ending of pre-ghost nodes
    LI m_uiDofLocalBegin; // local dof-ID starting of nodes owned by me
    LI m_uiDofLocalEnd; // local dof-ID ending of nodes owned by me
    LI m_uiDofPostGhostBegin; // local dof-ID starting of post-ghost nodes
    LI m_uiDofPostGhostEnd; // local dof-ID ending of post-ghost nodes

    std::vector<LI> m_uivIndependentElem; // Id of independent elements, i.e. elements do not have ghost dofs
    std::vector<LI> m_uivDependentElem; // Id of dependent elements

public:
    /**@brief constructor */
    Maps(MPI_Comm comm);

    /**@brief destructor */
    ~Maps();

    MPI_Comm get_comm() const
    {
        return m_comm;
    }

    /**@brief return variables without creating copies of variables, and not modifying */
    const LI& get_NumDofs() const
    {
        return m_uiNumDofs;
    }
    const LI& get_NumDofsTotal() const
    {
        return m_uiNumDofsTotal;
    }
    const GI& get_GlobalDofStart() const
    {
        return m_ulGlobalDofStart;
    }
    const GI& get_GlobalDofEnd() const
    {
        return m_ulGlobalDofEnd;
    }
    const GI& get_NumDofsGlobal() const
    {
        return m_ulNumDofsGlobal;
    }
    const LI& get_NumElems() const
    {
        return m_uiNumElems;
    }

    GI** get_Map() const { return m_ulpMap; }
    const LI* get_DofsPerElem() const
    {
        return m_uiDofsPerElem;
    }
    unsigned int** get_BdrMap() const
    {
        return m_uipBdrMap;
    }
    DT** get_PresValMap() const
    {
        return m_dtPresValMap;
    }

    const std::vector<GI>& get_ownedConstrainedDofs() const
    {
        return ownedConstrainedDofs;
    }
    const std::vector<DT>& get_ownedPrescribedValues() const
    {
        return ownedPrescribedValues;
    }
    const std::vector<GI>& get_ownedFreeDofs() const
    {
        return ownedFreeDofs;
    }

    const LI& get_n_owned_constraints() const
    {
        return n_owned_constraints;
    }

    LI** get_LocalMap() const
    {
        return m_uipLocalMap;
    }
    GI* get_Local2Global() const
    {
        return m_ulpLocal2Global;
    }

    const std::vector<LI>& get_LocalDofCounts() const
    {
        return m_uivLocalDofCounts;
    }
    const std::vector<LI>& get_LocalElementCounts() const
    {
        return m_uivLocalElementCounts;
    }
    const std::vector<GI>& get_LocalDofScan() const
    {
        return m_ulvLocalDofScan;
    }
    const std::vector<GI>& get_LocalElementScan() const
    {
        return m_ulvLocalElementScan;
    }

    const LI& get_NumPreGhostDofs() const
    {
        return m_uiNumPreGhostDofs;
    }
    const LI& get_NumPostGhostDofs() const
    {
        return m_uiNumPostGhostDofs;
    }

    const std::vector<LI>& get_SendDofCounts() const
    {
        return m_uivSendDofCounts;
    }
    const std::vector<LI>& get_SendDofOffset() const
    {
        return m_uivSendDofOffset;
    }
    const std::vector<LI>& get_SendDofIds() const
    {
        return m_uivSendDofIds;
    }
    const std::vector<unsigned int>& get_SendRankIds() const
    {
        return m_uivSendRankIds;
    }
    const std::vector<LI>& get_RecvDofCounts() const
    {
        return m_uivRecvDofCounts;
    }
    const std::vector<LI>& get_RecvDofOffset() const
    {
        return m_uivRecvDofOffset;
    }
    const std::vector<unsigned int>& get_RecvRankIds() const
    {
        return m_uivRecvRankIds;
    }

    const LI& get_DofPreGhostBegin() const
    {
        return m_uiDofPreGhostBegin;
    }
    const LI& get_DofPreGhostEnd() const
    {
        return m_uiDofPreGhostEnd;
    }
    const LI& get_DofLocalBegin() const
    {
        return m_uiDofLocalBegin;
    }
    const LI& get_DofLocalEnd() const
    {
        return m_uiDofLocalEnd;
    }
    const LI& get_DofPostGhostBegin() const
    {
        return m_uiDofPostGhostBegin;
    }
    const LI& get_DofPostGhostEnd() const
    {
        return m_uiDofPostGhostEnd;
    }
    const std::vector<LI>& get_independentElem() const
    {
        return m_uivIndependentElem;
    }
    const std::vector<LI>& get_dependentElem() const
    {
        return m_uivDependentElem;
    }

    /**@brief set mapping from element local node to global node */
    Error set_map(const LI n_elements_on_rank,
        const LI* const* element_to_rank_map,
        const LI* dofs_per_element,
        const LI n_all_dofs_on_rank, // Note: includes ghost dofs
        const GI* rank_to_global_map,
        const GI owned_global_dof_range_begin,
        const GI owned_global_dof_range_end,
        const GI n_global_dofs);

    /**@brief update map when cracks created */
    Error update_map(const LI* new_to_old_rank_map,
        const LI old_n_all_dofs_on_rank,
        const GI* old_rank_to_global_map,
        const LI n_elements_on_rank,
        const LI* const* element_to_rank_map,
        const LI* dofs_per_element,
        const LI n_all_dofs_on_rank,
        const GI* rank_to_global_map,
        const GI owned_global_dof_range_begin,
        const GI owned_global_dof_range_end,
        const GI n_global_dofs);

    /**@brief set boundary data, numConstraints is the global number of constrains */
    Error set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints);

    // ================ methods brough from aMatFree ======================
    /**@brief build scatter-gather map (used for communication) and local-to-local map (used for matvec) */
    Error buildScatterMap();

    /**@brief return true if DoF "enid" of element "eid" is owned by this rank, false otherwise */
    bool is_local_node(LI eid, LI enid) const
    {
        const LI nid = m_uipLocalMap[eid][enid];
        if (nid >= m_uiDofLocalBegin && nid < m_uiDofLocalEnd) {
            return true;
        } else {
            return false;
        }
    }

    /**@brief return the rank who owns gId */
    unsigned int globalId_2_rank(GI gId) const;

    /**@brief identify independent/dependent elements */
    Error identifyIndependentElements();

}; //class Maps

template <typename DT, typename GI, typename LI>
Maps<DT, GI, LI>::Maps(MPI_Comm comm)
{
    //m_comm             = MPI_COMM_NULL;
    m_comm = comm;
    MPI_Comm_rank(comm, (int*)&m_uiRank);
    MPI_Comm_size(comm, (int*)&m_uiSize);

    m_uiNumDofs = 0; // number of owned dofs
    m_uiNumDofsTotal = 0; // total number of owned dofs + ghost dofs
    m_ulGlobalDofStart = 0; // start of global dof id that I own
    m_ulGlobalDofEnd = 0; // end of global dof id that I own
    m_ulNumDofsGlobal = 0; // total number of dofs of all ranks
    m_uiNumElems = 0; // number of owned elements
    m_ulpMap = nullptr; // element-to-global map
    m_uiDofsPerElem = nullptr; // number of dofs per element
    m_uipBdrMap = nullptr; // element-to-constraint_flag map (1 is constrained dof, 0 is free dof)
    m_dtPresValMap = nullptr; // element-to-prescibed_value map
    m_uipLocalMap = nullptr; // element-to-local map
    n_owned_constraints = 0;
} // constructor

template <typename DT, typename GI, typename LI>
Maps<DT, GI, LI>::~Maps()
{

    // free element-to-global map
    if (m_ulpMap != nullptr) {
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            if (m_ulpMap[eid] != nullptr) {
                delete[] m_ulpMap[eid];
            }
        }
        delete[] m_ulpMap;
    }

    // free element-to-constraint_flag map
    if (m_uipBdrMap != nullptr) {
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            if (m_uipBdrMap[eid] != nullptr)
                delete[] m_uipBdrMap[eid];
        }
        delete[] m_uipBdrMap;
    }

    // free element-to-prescribed_value map
    if (m_dtPresValMap != nullptr) {
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            if (m_dtPresValMap[eid] != nullptr)
                delete[] m_dtPresValMap[eid];
        }
        delete[] m_dtPresValMap;
    }

    // free element-to-local map
    if (m_uipLocalMap != nullptr) {
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            if (m_uipLocalMap[eid] != nullptr) {
                delete[] m_uipLocalMap[eid];
            }
        }
        delete[] m_uipLocalMap;
    }

    // free local-to-global map (allocated in buildScatterMap)
    if (m_ulpLocal2Global != nullptr) {
        delete[] m_ulpLocal2Global;
    }

} // destructor

template <typename DT, typename GI, typename LI>
Error Maps<DT, GI, LI>::set_map(const LI n_elements_on_rank,
    const LI* const* element_to_rank_map,
    const LI* dofs_per_element,
    const LI n_all_dofs_on_rank,
    const GI* rank_to_global_map,
    const GI owned_global_dof_range_begin,
    const GI owned_global_dof_range_end,
    const GI n_global_dofs)
{
    // number of owned elements
    m_uiNumElems = n_elements_on_rank;

    // number of owned dofs
    m_uiNumDofs = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

    // number of dofs of ALL ranks, currently this is only used in aMatFree::petsc_dump_mat()
    m_ulNumDofsGlobal = n_global_dofs;

    // these are assertion in buildScatterMap
    m_ulGlobalDofStart = owned_global_dof_range_begin;
    m_ulGlobalDofEnd = owned_global_dof_range_end;
    m_uiNumDofsTotal = n_all_dofs_on_rank;

    // point to provided array giving number of dofs of each element
    m_uiDofsPerElem = dofs_per_element;

    // create global map based on provided local map and Local2Global
    m_ulpMap = new GI*[m_uiNumElems];
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        m_ulpMap[eid] = new GI[m_uiDofsPerElem[eid]];
    }
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++) {
            m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
        }
    }

    // 05.21.20: create local map that will be built in buildScatterMap
    m_uipLocalMap = new LI*[m_uiNumElems];
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        m_uipLocalMap[eid] = new LI[m_uiDofsPerElem[eid]];
    }

    // build scatter map for communication before and after matvec
    // 05.21.20: also build m_uipLocalMap
    buildScatterMap();

    // identify independent vs dependent elements so that overlap could be applied
    identifyIndependentElements();

    return Error::SUCCESS;

} // set_map

template <typename DT, typename GI, typename LI>
Error Maps<DT, GI, LI>::update_map(const LI* new_to_old_rank_map,
    const LI old_n_all_dofs_on_rank,
    const GI* old_rank_to_global_map,
    const LI n_elements_on_rank,
    const LI* const* element_to_rank_map,
    const LI* dofs_per_element,
    const LI n_all_dofs_on_rank,
    const GI* rank_to_global_map,
    const GI owned_global_dof_range_begin,
    const GI owned_global_dof_range_end,
    const GI n_global_dofs)
{

    // Number of owned elements should not be changed (extra dofs are enriched)
    assert(m_uiNumElems == n_elements_on_rank);

    // point to new provided array giving number of dofs of each element
    m_uiDofsPerElem = dofs_per_element;

    // delete the current global map in order to increase the size of 2nd dimension (i.e. number of dofs per element)
    if (m_ulpMap != nullptr) {
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            delete[] m_ulpMap[eid];
        }
    }
    // reallocate according to the new number of dofs per element
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        m_ulpMap[eid] = new GI[m_uiDofsPerElem[eid]];
    }
    // new global map
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++) {
            m_ulpMap[eid][nid] = rank_to_global_map[element_to_rank_map[eid][nid]];
        }
    }

    // 2020.05.21: delete the current local map in order to increase the size of 2nd dimension (i.e. number of dofs per element)
    if (m_uipLocalMap != nullptr) {
        for (LI eid = 0; eid < m_uiNumElems; eid++) {
            delete[] m_uipLocalMap[eid];
        }
    }
    // reallocate according to the new number of dofs per element
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        m_uipLocalMap[eid] = new LI[m_uiDofsPerElem[eid]];
    }

    // update number of owned dofs
    m_uiNumDofs = owned_global_dof_range_end - owned_global_dof_range_begin + 1;

    // update total dofs of all ranks, currently is only used by aMatFree::petsc_dump_mat()
    m_ulNumDofsGlobal = n_global_dofs;

    // update variables for assertion in buildScatterMap
    m_ulGlobalDofStart = owned_global_dof_range_begin;
    m_ulGlobalDofEnd = owned_global_dof_range_end;
    m_uiNumDofsTotal = n_all_dofs_on_rank;

    // build scatter map
    buildScatterMap();

    return Error::SUCCESS;

} // update_map()

template <typename DT, typename GI, typename LI>
Error Maps<DT, GI, LI>::buildScatterMap()
{
    /* Assumptions: We assume that the global nodes are continuously partitioned across processors. */

    // save the total dofs (owned + ghost) provided in set_map for assertion
    // m_uiNumDofsTotal will be re-computed based on: m_ulpMap, m_uiNumDofs, m_uiNumElems
    const LI m_uiNumDofsTotal_received_in_setmap = m_uiNumDofsTotal;

    // save the global number of dofs (of all ranks) provided in set_map for assertion
    // m_ulNumDofsGlobal will be re-computed based on: m_uiNumDofs
    const GI m_ulNumDofsGlobal_received_in_setmap = m_ulNumDofsGlobal;

    if (m_ulpMap == nullptr) {
        return Error::NULL_L2G_MAP;
    }

    m_uivLocalDofCounts.clear();
    m_uivLocalElementCounts.clear();
    m_ulvLocalDofScan.clear();
    m_ulvLocalElementScan.clear();

    m_uivLocalDofCounts.resize(m_uiSize);
    m_uivLocalElementCounts.resize(m_uiSize);
    m_ulvLocalDofScan.resize(m_uiSize);
    m_ulvLocalElementScan.resize(m_uiSize);

    // gather local counts
    MPI_Allgather(&m_uiNumDofs, 1, MPI_INT, m_uivLocalDofCounts.data(), 1, MPI_INT, m_comm);
    MPI_Allgather(&m_uiNumElems, 1, MPI_INT, m_uivLocalElementCounts.data(), 1, MPI_INT, m_comm);

    // scan local counts to determine owned-range:
    // range of global ID of owned dofs = [m_ulvLocalDofScan[m_uiRank], m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)
    m_ulvLocalDofScan[0] = 0;
    m_ulvLocalElementScan[0] = 0;
    for (unsigned int p = 1; p < m_uiSize; p++) {
        m_ulvLocalDofScan[p] = m_ulvLocalDofScan[p - 1] + m_uivLocalDofCounts[p - 1];
        m_ulvLocalElementScan[p] = m_ulvLocalElementScan[p - 1] + m_uivLocalElementCounts[p - 1];
    }

    // global number of dofs of all ranks
    m_ulNumDofsGlobal = m_ulvLocalDofScan[m_uiSize - 1] + m_uivLocalDofCounts[m_uiSize - 1];
    assert(m_ulNumDofsGlobal == m_ulNumDofsGlobal_received_in_setmap);

    // dofs are not owned by me: stored in pre or post lists
    std::vector<GI> preGhostGIds;
    std::vector<GI> postGhostGIds;
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        for (LI i = 0; i < m_uiDofsPerElem[eid]; i++) {
            // global ID
            const GI global_dof_id = m_ulpMap[eid][i];
            if (global_dof_id < m_ulvLocalDofScan[m_uiRank]) {
                // dofs with global ID < owned-range --> pre-ghost dofs
                assert(global_dof_id < m_ulGlobalDofStart); // m_ulGlobalDofStart was passed in set_map
                preGhostGIds.push_back(global_dof_id);
            } else if (global_dof_id >= (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
                // dofs with global ID > owned-range --> post-ghost dofs
                // note: m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs - 1 = m_ulGlobalDofEnd
                assert(global_dof_id > m_ulGlobalDofEnd); // m_ulGlobalDofEnd was passed in set_map
                postGhostGIds.push_back(global_dof_id);
            } else {
                assert((global_dof_id >= m_ulvLocalDofScan[m_uiRank])
                    && (global_dof_id < (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)));
                assert((global_dof_id >= m_ulGlobalDofStart) && (global_dof_id <= m_ulGlobalDofEnd));
            }
        }
    }

    // sort in ascending order
    std::sort(preGhostGIds.begin(), preGhostGIds.end());
    std::sort(postGhostGIds.begin(), postGhostGIds.end());

    // remove consecutive duplicates and erase all after .end()
    preGhostGIds.erase(std::unique(preGhostGIds.begin(), preGhostGIds.end()), preGhostGIds.end());
    postGhostGIds.erase(std::unique(postGhostGIds.begin(), postGhostGIds.end()), postGhostGIds.end());

    // number of pre and post ghost dofs
    m_uiNumPreGhostDofs = static_cast<LI>(preGhostGIds.size());
    m_uiNumPostGhostDofs = static_cast<LI>(postGhostGIds.size());

    // range of local ID of pre-ghost dofs = [0, m_uiDofPreGhostEnd)
    m_uiDofPreGhostBegin = 0;
    m_uiDofPreGhostEnd = m_uiNumPreGhostDofs;

    // range of local ID of owned dofs = [m_uiDofLocalBegin, m_uiDofLocalEnd)
    m_uiDofLocalBegin = m_uiDofPreGhostEnd;
    m_uiDofLocalEnd = m_uiDofLocalBegin + m_uiNumDofs;

    // range of local ID of post-ghost dofs = [m_uiDofPostGhostBegin, m_uiDofPostGhostEnd)
    m_uiDofPostGhostBegin = m_uiDofLocalEnd;
    m_uiDofPostGhostEnd = m_uiDofPostGhostBegin + m_uiNumPostGhostDofs;

    // total number of dofs including ghost dofs
    m_uiNumDofsTotal = m_uiNumDofs + m_uiNumPreGhostDofs + m_uiNumPostGhostDofs;

    // note: m_uiNumDofsTotal was passed in set_map --> assert if what we received is what we compute here
    assert(m_uiNumDofsTotal == m_uiNumDofsTotal_received_in_setmap);

    // determine owners of pre- and post-ghost dofs
    std::vector<unsigned int> preGhostOwner;
    std::vector<unsigned int> postGhostOwner;
    preGhostOwner.resize(m_uiNumPreGhostDofs);
    postGhostOwner.resize(m_uiNumPostGhostDofs);

    // pre-ghost
    unsigned int pcount = 0; // processor counter, start from 0
    LI gcount = 0; // counter of ghost dof
    while (gcount < m_uiNumPreGhostDofs) {
        // global ID of pre-ghost dof gcount
        GI global_dof_id = preGhostGIds[gcount];
        while ((pcount < m_uiRank) && (!((global_dof_id >= m_ulvLocalDofScan[pcount]) && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount]))))) {
            // global_dof_id is not owned by pcount
            pcount++;
        }
        // check if global_dof_id is really in the range of global ID of dofs owned by pcount
        if (!((global_dof_id >= m_ulvLocalDofScan[pcount])
                && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount])))) {
            std::cout << "m_uiRank: " << m_uiRank << " pre ghost gid : " << global_dof_id << " was not found in any processor" << std::endl;
            return Error::GHOST_NODE_NOT_FOUND;
        }
        preGhostOwner[gcount] = pcount;
        gcount++;
    }

    // post-ghost
    pcount = m_uiRank; // processor counter, start from my rank
    gcount = 0;
    while (gcount < m_uiNumPostGhostDofs) {
        // global ID of post-ghost dof gcount
        GI global_dof_id = postGhostGIds[gcount];
        while ((pcount < m_uiSize) && (!((global_dof_id >= m_ulvLocalDofScan[pcount]) && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount]))))) {
            // global_dof_id is not owned by pcount
            pcount++;
        }
        // check if global_dof_id is really in the range of global ID of dofs owned by pcount
        if (!((global_dof_id >= m_ulvLocalDofScan[pcount])
                && (global_dof_id < (m_ulvLocalDofScan[pcount] + m_uivLocalDofCounts[pcount])))) {
            std::cout << "m_uiRank: " << m_uiRank << " post ghost gid : " << global_dof_id << " was not found in any processor" << std::endl;
            return Error::GHOST_NODE_NOT_FOUND;
        }
        postGhostOwner[gcount] = pcount;
        gcount++;
    }

    LI* sendCounts = new LI[m_uiSize];
    LI* recvCounts = new LI[m_uiSize];
    LI* sendOffset = new LI[m_uiSize];
    LI* recvOffset = new LI[m_uiSize];

    // Note: the send here is just for use in MPI_Alltoallv, it is NOT the send in communications between processors later
    for (unsigned int i = 0; i < m_uiSize; i++) {
        // many of these will be zero, only non zero for processors that own my ghost nodes
        sendCounts[i] = 0;
    }

    // count number of pre-ghost dofs to corresponding owners
    for (LI i = 0; i < m_uiNumPreGhostDofs; i++) {
        // preGhostOwner[i] = rank who owns the ith pre-ghost dof
        sendCounts[preGhostOwner[i]] += 1;
    }

    // count number of post-ghost dofs to corresponding owners
    for (LI i = 0; i < m_uiNumPostGhostDofs; i++) {
        // postGhostOwner[i] = rank who owns the ith post-ghost dof
        sendCounts[postGhostOwner[i]] += 1;
    }

    // get recvCounts by transposing the matrix of sendCounts
    MPI_Alltoall(sendCounts, 1, MPI_UNSIGNED, recvCounts, 1, MPI_UNSIGNED, m_comm);

    // compute offsets from sends
    sendOffset[0] = 0;
    recvOffset[0] = 0;
    for (unsigned int i = 1; i < m_uiSize; i++) {
        sendOffset[i] = sendOffset[i - 1] + sendCounts[i - 1];
        recvOffset[i] = recvOffset[i - 1] + recvCounts[i - 1];
    }

    // size of sendBuf = # ghost dofs (i.e. # dofs I need but not own)
    // later, this is used as number of dofs that I need to receive from corresponding rank before doing matvec
    std::vector<GI> sendBuf;
    sendBuf.resize(sendOffset[m_uiSize - 1] + sendCounts[m_uiSize - 1]); //size also = (m_uiNumPreGhostDofs + m_uiNumPostGhostDofs)

    // size of recvBuf = sum of dofs each other rank needs from me
    // later, this is used as the number of dofs that I need to send to ranks (before doing matvec) who need them as ghost dofs
    std::vector<GI> recvBuf;
    recvBuf.resize(recvOffset[m_uiSize - 1] + recvCounts[m_uiSize - 1]);

    // put global ID of pre- and post-ghost dofs to sendBuf
    for (LI i = 0; i < m_uiNumPreGhostDofs; i++)
        sendBuf[i] = preGhostGIds[i];
    for (LI i = 0; i < m_uiNumPostGhostDofs; i++)
        sendBuf[i + m_uiNumPreGhostDofs] = postGhostGIds[i];

    for (unsigned int i = 0; i < m_uiSize; i++) {
        sendCounts[i] *= sizeof(GI);
        sendOffset[i] *= sizeof(GI);
        recvCounts[i] *= sizeof(GI);
        recvOffset[i] *= sizeof(GI);
    }

    // exchange the global ID of ghost dofs with ranks who own them
    MPI_Alltoallv(sendBuf.data(), (int*)sendCounts, (int*)sendOffset, MPI_BYTE,
        recvBuf.data(), (int*)recvCounts, (int*)recvOffset, MPI_BYTE, m_comm);

    for (unsigned int i = 0; i < m_uiSize; i++) {
        sendCounts[i] /= sizeof(GI);
        sendOffset[i] /= sizeof(GI);
        recvCounts[i] /= sizeof(GI);
        recvOffset[i] /= sizeof(GI);
    }

    // convert global Ids in recvBuf (dofs that I need to send to before matvec) to local Ids
    m_uivSendDofIds.resize(recvBuf.size());

    for (LI i = 0; i < recvBuf.size(); i++) {
        // global ID of recvBuf[i]
        const GI global_dof_id = recvBuf[i];
        // check if global_dof_id is really owned by my rank, if not then something went wrong with sendBuf above
        if (global_dof_id < m_ulvLocalDofScan[m_uiRank] || global_dof_id >= (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
            std::cout << " m_uiRank: " << m_uiRank << "scatter map error : " << __func__ << std::endl;
            Error::GHOST_NODE_NOT_FOUND;
        }
        // also check with data passed in set_map
        assert((global_dof_id >= m_ulGlobalDofStart) && (global_dof_id <= m_ulGlobalDofEnd));
        // convert global id to local id (id local to rank)
        m_uivSendDofIds[i] = m_uiNumPreGhostDofs + (global_dof_id - m_ulvLocalDofScan[m_uiRank]);
    }

    m_uivSendDofCounts.resize(m_uiSize);
    m_uivSendDofOffset.resize(m_uiSize);
    m_uivRecvDofCounts.resize(m_uiSize);
    m_uivRecvDofOffset.resize(m_uiSize);

    for (unsigned int i = 0; i < m_uiSize; i++) {
        m_uivSendDofCounts[i] = recvCounts[i];
        m_uivSendDofOffset[i] = recvOffset[i];
        m_uivRecvDofCounts[i] = sendCounts[i];
        m_uivRecvDofOffset[i] = sendOffset[i];
    }

    // identify ranks that I need to send to and ranks that I will receive from
    for (unsigned int i = 0; i < m_uiSize; i++) {
        if (m_uivSendDofCounts[i] > 0) {
            m_uivSendRankIds.push_back(i);
        }
        if (m_uivRecvDofCounts[i] > 0) {
            m_uivRecvRankIds.push_back(i);
        }
    }

    // 2020.05.21: build rank-to-global map and element-to-rank map
    // local vector = [0, ..., (m_uiNumPreGhostDofs - 1), --> ghost nodes owned by someone before me
    // m_uiNumPreGhostDofs, ..., (m_uiNumPreGhostDofs + m_uiNumDofs - 1), --> nodes owned by me
    // (m_uiNumPreGhostDofs + m_uiNumDofs), ..., (m_uiNumPreGhostDofs + m_uiNumDofs + m_uiNumPostGhostDofs - 1)] --> nodes owned by someone after me
    m_ulpLocal2Global = new GI[m_uiNumDofsTotal];
    LI local_dof_id;
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        for (LI i = 0; i < m_uiDofsPerElem[eid]; i++) {
            // global Id of i
            const GI global_dof_id = m_ulpMap[eid][i];
            if (global_dof_id >= m_ulvLocalDofScan[m_uiRank] && global_dof_id < (m_ulvLocalDofScan[m_uiRank] + m_uivLocalDofCounts[m_uiRank])) {
                // global_dof_id is owned by me
                local_dof_id = global_dof_id - m_ulvLocalDofScan[m_uiRank] + m_uiNumPreGhostDofs;

            } else if (global_dof_id < m_ulvLocalDofScan[m_uiRank]) {
                // global_dof_id is owned by someone before me (note: can be safely cast to LI due to size of preGhostGIds)
                const LI lookUp = static_cast<LI>(std::lower_bound(preGhostGIds.begin(), preGhostGIds.end(), global_dof_id) - preGhostGIds.begin());
                local_dof_id = lookUp;

            } else if (global_dof_id >= (m_ulvLocalDofScan[m_uiRank] + m_uivLocalDofCounts[m_uiRank])) {
                // global_dof_id is owned by someone after me (note: can be safely cast to LI due to size of preGhostGIds)
                const LI lookUp = static_cast<LI>(std::lower_bound(postGhostGIds.begin(), postGhostGIds.end(), global_dof_id) - postGhostGIds.begin());
                local_dof_id = (m_uiNumPreGhostDofs + m_uiNumDofs) + lookUp;
            } else {
                std::cout << " m_uiRank: " << m_uiRank << "scatter map error : " << __func__ << std::endl;
                Error::GLOBAL_DOF_ID_NOT_FOUND;
            }
            m_uipLocalMap[eid][i] = local_dof_id;
            m_ulpLocal2Global[local_dof_id] = global_dof_id;
        }
    }

    delete[] sendCounts;
    delete[] recvCounts;
    delete[] sendOffset;
    delete[] recvOffset;

    return Error::SUCCESS;

} // buildScatterMap()

template <typename DT, typename GI, typename LI>
Error Maps<DT, GI, LI>::set_bdr_map(GI* constrainedDofs, DT* prescribedValues, LI numConstraints)
{

    // extract constrained dofs owned by me
    for (LI i = 0; i < numConstraints; i++) {
        auto global_Id = constrainedDofs[i];
        if ((global_Id >= m_ulvLocalDofScan[m_uiRank]) && (global_Id < m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
            ownedConstrainedDofs.push_back(global_Id);
            ownedPrescribedValues.push_back(prescribedValues[i]);
        }
    }

    // construct elemental map of boundary condition
    m_uipBdrMap = new unsigned int*[m_uiNumElems];
    m_dtPresValMap = new DT*[m_uiNumElems];
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        m_uipBdrMap[eid] = new unsigned int[m_uiDofsPerElem[eid]];
        m_dtPresValMap[eid] = new DT[m_uiDofsPerElem[eid]];
    }

    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        for (LI nid = 0; nid < m_uiDofsPerElem[eid]; nid++) {
            auto global_Id = m_ulpMap[eid][nid];
            LI index;
            for (index = 0; index < numConstraints; index++) {
                if (global_Id == constrainedDofs[index]) {
                    m_uipBdrMap[eid][nid] = 1;
                    m_dtPresValMap[eid][nid] = prescribedValues[index];
                    break;
                }
            }
            if (index == numConstraints) {
                m_uipBdrMap[eid][nid] = 0;
                m_dtPresValMap[eid][nid] = -1E16; // for testing
                if ((global_Id >= m_ulvLocalDofScan[m_uiRank]) && (global_Id < m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs)) {
                    ownedFreeDofs.push_back(global_Id);
                }
            }
        }
    }

    //if (ownedFreeDofs_unsorted.size() > 0){
    if (ownedFreeDofs.size() > 0) {
        std::sort(ownedFreeDofs.begin(), ownedFreeDofs.end());
        ownedFreeDofs.erase(std::unique(ownedFreeDofs.begin(), ownedFreeDofs.end()), ownedFreeDofs.end());
    }

    // number of owned constraints
    n_owned_constraints = static_cast<LI>(ownedConstrainedDofs.size());

    return Error::SUCCESS;
} // set_bdr_map

// return rank that owns global gId
template <typename DT, typename GI, typename LI>
unsigned int Maps<DT, GI, LI>::globalId_2_rank(GI gId) const
{
    LI rank;
    auto it =std::lower_bound(m_ulvLocalDofScan.begin(),m_ulvLocalDofScan.end(),gId);
    if(it==m_ulvLocalDofScan.end())
    {
        assert(gId>=m_ulvLocalDofScan[m_uiSize - 1]);
        rank = m_uiSize - 1;
    }else
    {
        rank=std::distance(m_ulvLocalDofScan.begin(), it);
        if(gId < m_ulvLocalDofScan[rank])
        {
            assert(rank>0);
            rank=rank-1;
        }

    }
    // this is a performance killer. 
    // unsigned int rank;
    // if (gId >= m_ulvLocalDofScan[m_uiSize - 1]) {
    //     rank = m_uiSize - 1;
    // } else {
    //     for (unsigned int i = 0; i < (m_uiSize - 1); i++) {
    //         if (gId >= m_ulvLocalDofScan[i] && gId < m_ulvLocalDofScan[i + 1] && (i < (m_uiSize - 1))) {
    //             rank = i;
    //             break;
    //         }
    //     }
    // }
    return rank;

} // globalId_2_rank

template <typename DT, typename GI, typename LI>
Error Maps<DT, GI, LI>::identifyIndependentElements()
{
    m_uivDependentElem.clear();
    m_uivIndependentElem.clear();

    GI global_dof_id;
    LI did;
    for (LI eid = 0; eid < m_uiNumElems; eid++) {
        for (did = 0; did < m_uiDofsPerElem[eid]; did++) {
            // global dof id
            global_dof_id = m_ulpMap[eid][did];
            // eid has at least 1 ghost dof --> dependent
            if ((global_dof_id < m_ulvLocalDofScan[m_uiRank]) || (global_dof_id >= (m_ulvLocalDofScan[m_uiRank] + m_uiNumDofs))) {
                m_uivDependentElem.push_back(eid);
                break;
            }
        }
        // eid does not have ghost dof --> independent
        if (did == m_uiDofsPerElem[eid]) {
            m_uivIndependentElem.push_back(eid);
        }
    }
    assert((m_uivIndependentElem.size() + m_uivDependentElem.size()) == m_uiNumElems);
    /* for (LI eid = 0; eid < m_uivDependentElem.size(); eid++){
            printf("[r%d], dependentE[%d]= %d\n", m_uiRank, eid, m_uivDependentElem[eid]);
        }
        for (LI eid = 0; eid < m_uivIndependentElem.size(); eid++){
            printf("[r%d], independentE[%d]= %d\n", m_uiRank, eid, m_uivIndependentElem[eid]);
        } */
    return Error::SUCCESS;
}

} //end of namespace par
#endif // APTIVEMATRIX_MAPS_H