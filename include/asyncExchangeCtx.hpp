/**
 * @file asyncExchangeCtx.hpp
 * @author Hari Sundar      hsundar@gmail.com
 * @author Milinda Fernando milinda@cs.utah.edu
 *
 * @brief Class for data exchange among MPI processes
 * 
 * @version
 * @date
 * 
 * @copyright Copyright (c) 2018 School of Computing, University of Utah
 * 
 */

#ifndef ADAPTIVEMATRIX_ASYNCEXCHANGECTX_H
#define ADAPTIVEMATRIX_ASYNCEXCHANGECTX_H

#include <mpi.h>
#include <vector>

namespace par {
// Class AsyncExchangeCtx is downloaded from Dendro-5.0 with permission from the author (Milinda Fernando)
// Dendro-5.0 is written by Milinda Fernando and Hari Sundar;
class AsyncExchangeCtx {
private:
    /** pointer to the variable which perform the ghost exchange */
    void* m_uiBuffer;

    /** pointer to the send buffer*/
    void* m_uiSendBuf;

    /** pointer to the receive buffer*/
    void* m_uiRecvBuf;

    /** list of request*/
    std::vector<MPI_Request*> m_uiRequests;

public:
    /**@brief creates an async ghost exchange context*/
    AsyncExchangeCtx(const void* var)
    {
        m_uiBuffer = (void*)var;
        m_uiSendBuf = nullptr;
        m_uiRecvBuf = nullptr;
        m_uiRequests.clear();
    }

    /**@brief allocates send buffer for ghost exchange */
    void allocateSendBuffer(size_t bytes) { m_uiSendBuf = malloc(bytes); }

    /**@brief allocates recv buffer for ghost exchange */
    void allocateRecvBuffer(size_t bytes) { m_uiRecvBuf = malloc(bytes); }

    /**@brief allocates send buffer for ghost exchange */
    void deAllocateSendBuffer()
    {
        free(m_uiSendBuf);
        m_uiSendBuf = nullptr;
    }

    /**@brief allocates recv buffer for ghost exchange */
    void deAllocateRecvBuffer()
    {
        free(m_uiRecvBuf);
        m_uiRecvBuf = nullptr;
    }

    /**@brief */
    void* getSendBuffer() { return m_uiSendBuf; }

    /**@brief */
    void* getRecvBuffer() { return m_uiRecvBuf; }

    /**@brief */
    const void* getBuffer() { return m_uiBuffer; }

    /**@brief */
    std::vector<MPI_Request*>& getRequestList() { return m_uiRequests; }

    /**@brief */
    bool operator==(AsyncExchangeCtx other) const { return (m_uiBuffer == other.m_uiBuffer); }

    ~AsyncExchangeCtx()
    {
        /*for(unsigned int i=0;i<m_uiRequests.size();i++)
              {
              delete m_uiRequests[i];
              m_uiRequests[i]=nullptr;
              }
              m_uiRequests.clear();*/
    }
}; // class AsyncExchangeCtx

} // namespace par
#endif // APTIVEMATRIX_ASYNCEXCHANGECTX_H