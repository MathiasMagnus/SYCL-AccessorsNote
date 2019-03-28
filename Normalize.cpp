// SYCL includes
#include <CL/sycl.hpp>

// Standard C++ includes
#include <iostream>
#include <algorithm>
#include <random>

namespace cl::sycl::algo
{
    template <typename K, typename T1, typename T2, typename F, typename... Placeholders>
    cl::sycl::event transform(cl::sycl::queue q,
                              cl::sycl::buffer<T1, 1> buf1,
                              cl::sycl::buffer<T2, 1> buf2,
                              F f,
                              Placeholders... placeholders);
    
    template <typename K, typename T1, typename T2, typename F>
    void reduce(cl::sycl::queue q,
                cl::sycl::buffer<T1> buf1,
                cl::sycl::buffer<T2> buf2,
                F f);
}

// Implementation of algorithms
#include "TransformPlaceholder.hpp"
#include "Reduce.hpp"

namespace kernels
{
    struct obtain_maximum;
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

        auto prng = [engine = std::default_random_engine{ std::random_device{}() },
                     dist = std::uniform_real_distribution<real>{ 0.0f, 100.f }]() mutable { return dist(engine); };

        std::generate_n(r.get_pointer(), r.get_count(), prng);
    }

    std::cout << "Normalizing... "; std::cout.flush();
    // std::max<real()> takes by ref, mode::read accessors return by value, binding l-value ref to temporary results in error
    cl::sycl::algo::reduce<kernels::obtain_maximum>(q, rand, maxi,
                                                    [=](const real lhs, const real rhs) { return lhs > rhs ? lhs : rhs; });
    auto m = cl::sycl::accessor<real, 1,
                                cl::sycl::access::mode::read,
                                cl::sycl::access::target::constant_buffer,
                                cl::sycl::access::placeholder::true_t>{ maxi };
    auto normalize = [=](const real & val) { return val / m[0]; };
    cl::sycl::algo::transform<kernels::normalize>(q, rand, norm, normalize, m);

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
