//
// Created by milinda on 10/20/17.
/**
*@author Milinda Fernando
*School of Computing, University of Utah
*@brief simple profiler based on Hari's sort_profiler for bssn application.
*/
//

#ifndef SFCSORTBENCH_DENDRO_PROFILER_H
#define SFCSORTBENCH_DENDRO_PROFILER_H

#include "mpi.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#ifdef HAVE_PAPI
#include <papi.h>
#endif

#include <omp.h>

// Class profiler_t is downloaded from Dendro-5.0 with permission from the author (Milinda Fernando)
// Dendro-5.0 is written by Milinda Fernando and Hari Sundar;
class profiler_t {
public:
    long double seconds; // openmp wall time
    long long p_flpops; // papi floating point operations
    long double snap; // snap shot of the cumilative time.
    long long num_calls; // number of times the timer stop function called.

protected:
    long double _pri_seconds; // openmp wall time
    long long _pri_p_flpops; // papi floating point operations

public:
    profiler_t()
    {
        seconds = 0.0; // openmp wall time
        p_flpops = 0; // papi floating point operations
        snap = 0.0;
        num_calls = 0;

        _pri_seconds = 0.0;
        _pri_p_flpops = 0;
    }
    virtual ~profiler_t() { }

    void start()
    {
        _pri_seconds = omp_get_wtime();
        flops_papi();
    }
    void stop()
    {
        seconds -= _pri_seconds;
        p_flpops -= _pri_p_flpops;
        snap -= _pri_seconds;

        _pri_seconds = omp_get_wtime();
        flops_papi();

        seconds += _pri_seconds;
        p_flpops += _pri_p_flpops;
        snap += _pri_seconds;
        //num_calls++;
    }
    void clear()
    {
        seconds = 0.0;
        p_flpops = 0;
        snap = 0.0;
        num_calls = 0;

        _pri_seconds = 0.0;
        _pri_p_flpops = 0;
    }
    void snapreset()
    {
        snap = 0.0;
        num_calls = 0;
    }

private:
    void flops_papi()
    {
#ifdef HAVE_PAPI
        int retval;
        float rtime, ptime, mflops;
        retval = PAPI_flops(&rtime, &ptime, &_pri_p_flpops, &mflops);
        // assert (retval == PAPI_OK);
#else
        _pri_p_flpops = 0;
#endif
    }
};

#endif //SFCSORTBENCH_DENDRO_PROFILER_H
