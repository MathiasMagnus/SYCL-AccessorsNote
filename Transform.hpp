// SYCL includes
#include <CL/sycl.hpp>

// Standard C++ includes
#include <cassert>


namespace cl::sycl::algo
{
    template <typename K, typename T1, typename T2, typename F>
    cl::sycl::event transform(cl::sycl::queue q,
                              cl::sycl::buffer<T1, 1> buf1,
                              cl::sycl::buffer<T2, 1> buf2,
                              F f)
    {
        assert(buf1.get_range() == buf2.get_range());

        return q.submit([&](cl::sycl::handler & cgh)
        {
            auto b1 = buf1.template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::global_buffer>(cgh);
            auto b2 = buf2.template get_access<cl::sycl::access::mode::discard_write, cl::sycl::access::target::global_buffer>(cgh);

            cgh.parallel_for<K>(buf1.get_range(), [=](const cl::sycl::item<1> i)
            {
                b2[i] = f(b1[i]);
            });
        });
    }
}
