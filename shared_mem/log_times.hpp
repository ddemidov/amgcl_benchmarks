#ifndef CODE_LOG_TIMES_HPP
#define CODE_LOG_TIMES_HPP

#include <iostream>
#include <fstream>

inline void log_times(const char *fname, int np, int size, int iters, double setup, double solve) {
    std::ofstream f(fname, std::ios::app);
    f << np << " " << size << " " << iters << " "
        << std::scientific << setup << " " << solve << std::endl;
}

#endif
