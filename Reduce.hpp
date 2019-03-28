/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  reduction.cpp
 *
 *  Description:
 *    Example of a reduction operation in SYCL.
 *
 **************************************************************************/

// SYCL includes
#include <CL/sycl.hpp>

// Standard C++ includes
#include <cassert>


namespace cl::sycl::algo
{
    template <typename K, typename T1, typename T2, typename F>
    void reduce(cl::sycl::queue q,
                cl::sycl::buffer<T1> buf1,
                cl::sycl::buffer<T2> buf2,
                F f)
    {
        assert(buf1.get_range().size() != 0 &&
               buf2.get_range().size() != 0);

        cl::sycl::buffer<T1> bufI{ buf1.get_range() };

        q.submit([&](cl::sycl::handler & cgh)
        {
            auto from = buf1.template get_access<cl::sycl::access::mode::read>(cgh);
            auto to = bufI.template get_access<cl::sycl::access::mode::discard_write>(cgh);

            cgh.copy(from, to);
        });

        std::size_t length = bufI.get_range().size(),
                    local = std::min(bufI.get_range().size(),
                                     q.get_info<cl::sycl::info::queue::device>().get_info<cl::sycl::info::device::max_work_group_size>());
        do
        {
            auto step = [length, local, f, &bufI](cl::sycl::handler & h) mutable
            {
                cl::sycl::nd_range<1> r{ cl::sycl::range<1>{ std::max(length, local) },
                                         cl::sycl::range<1>{ std::min(length, local) } };

                auto aI = bufI.template get_access<cl::sycl::access::mode::read_write>(h);
                cl::sycl::accessor<T1, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local> scratch(cl::sycl::range<1>(local), h);

                h.parallel_for<K>(r, [aI, scratch, local, length, f](cl::sycl::nd_item<1> id)
                {
                    size_t globalid = id.get_global_id(0);
                    size_t localid = id.get_local_id(0);

                    if (globalid < length)
                    {
                        scratch[localid] = aI[globalid];
                    }
                    id.barrier(cl::sycl::access::fence_space::local_space);

                    if (globalid < length)
                    {
                        int min = cl::sycl::min(length, local);
                        for (size_t offset = min / 2; offset > 0; offset /= 2)
                        {
                            if (localid < offset)
                            {
                                scratch[localid] = f(scratch[localid], scratch[localid + offset]);
                            }
                            id.barrier(cl::sycl::access::fence_space::local_space);
                        }

                        if (localid == 0)
                        {
                            aI[id.get_group(0)] = scratch[localid];
                        }
                    }
                });
            };
            q.submit(step);

            length = length / local;
        } while (length > 1);

        q.submit([&](cl::sycl::handler & cgh)
        {
            auto from = bufI.template get_access<cl::sycl::access::mode::read>(cgh, cl::sycl::range<1>{1}, cl::sycl::id<1>{0});
            auto to = buf2.template get_access<cl::sycl::access::mode::write>(cgh, cl::sycl::range<1>{1}, cl::sycl::id<1>{0});

            cgh.copy(from, to);
        });
    }
}
