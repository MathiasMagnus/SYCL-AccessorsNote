// SYCL includes
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <algorithm>
#include <random>

namespace cl::sycl::algo
{
    template <typename K, typename T1, typename T2, typename F>
    cl::sycl::event transform(cl::sycl::queue q,
                              cl::sycl::buffer<T1, 1> buf1,
                              cl::sycl::buffer<T2, 1> buf2,
                              F f);
}

// Implementation of algorithms
#include "Transform.hpp"

namespace kernels
{
    struct normalize;
}

int main()
{
    using real = float;
    std::size_t length = 262144;

    cl::sycl::queue q;
    std::cout << "Using device: " << q.get_info<cl::sycl::info::queue::device>().get_info<cl::sycl::info::device::name>() << std::endl;

    cl::sycl::buffer<real> rand{ length },
                           norm{ length },
                           maxi{ 1 };
    {
        std::cout << "Generating " << length << " random numbers" << std::endl;
        auto r = rand.get_access<cl::sycl::access::mode::write>();
        auto m = maxi.get_access<cl::sycl::access::mode::write>();

        auto prng = [engine = std::default_random_engine{ std::random_device{}() },
                     dist = std::uniform_real_distribution<real>{ 0.0f, 100.f }]() mutable { return dist(engine); };

        std::generate_n(r.get_pointer(), r.get_count(), prng);
        m[0] = *std::max_element(r.get_pointer(), r.get_pointer() + r.get_count());
    }

    // This is what would be ideal, idiomatic SYCL.
    // Placeholder accessors are "magically" required, names of accessors don't leak.
    //
    // This currently crashes on submit.
    auto normalize = [m = cl::sycl::accessor<real, 1,
                                             cl::sycl::access::mode::read,
                                             cl::sycl::access::target::constant_buffer,
                                             cl::sycl::access::placeholder::true_t>{ maxi }](const real& val) { return val / m[0]; };
    cl::sycl::algo::transform<kernels::normalize>(q, rand, norm, normalize);

    {
        auto r = rand.get_access<cl::sycl::access::mode::read>();
        auto n = norm.get_access<cl::sycl::access::mode::read>();
        std::cout << "done!" << std::endl;

        std::cout.precision(12);
        for (auto i : { 1, 20, 300, 4000, 50000 })
            std::cout << r[i] << '\t' << n[i] << std::endl;
    }

    return 0;
}
