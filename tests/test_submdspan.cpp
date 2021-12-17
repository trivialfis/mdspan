/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <experimental/mdspan>
#include <vector>

#include <gtest/gtest.h>

namespace stdex = std::experimental;
_MDSPAN_INLINE_VARIABLE constexpr auto dyn = stdex::dynamic_extent;

TEST(TestSubmdspanLayoutRightStaticSizedRankReducing3Dto1D, test_submdspan_layout_right_static_sized_rank_reducing_3d_to_1d) {
  std::vector<int> d(2 * 3 * 4, 0);
  stdex::mdspan<int, stdex::extents<2, 3, 4>> m(d.data());
  __MDSPAN_OP(m, 1, 1, 1) = 42;
  auto sub0 = stdex::submdspan(m, 1, 1, stdex::full_extent);
  ASSERT_EQ(sub0.rank(),         1);
  ASSERT_EQ(sub0.rank_dynamic(), 0);
  ASSERT_EQ(sub0.extent(0),      4);
  ASSERT_EQ((__MDSPAN_OP(sub0, 1)), 42);
}

TEST(TestSubmdspanLayoutLeftStaticSizedRankReducing3Dto1D, test_submdspan_layout_left_static_sized_rank_reducing_3d_to_1d) {
  std::vector<int> d(2 * 3 * 4, 0);
  stdex::mdspan<int, stdex::extents<2, 3, 4>, stdex::layout_left> m(d.data());
  __MDSPAN_OP(m, 1, 1, 1) = 42;
  auto sub0 = stdex::submdspan(m, 1, 1, stdex::full_extent);
  ASSERT_EQ(sub0.rank(),         1);
  ASSERT_EQ(sub0.rank_dynamic(), 0);
  ASSERT_EQ(sub0.extent(0),      4);
  ASSERT_EQ((__MDSPAN_OP(sub0, 1)), 42);
}

TEST(TestSubmdspanLayoutRightStaticSizedRankReducingNested3Dto0D, test_submdspan_layout_right_static_sized_rank_reducing_nested_3d_to_0d) {
  std::vector<int> d(2 * 3 * 4, 0);
  stdex::mdspan<int, stdex::extents<2, 3, 4>> m(d.data());
  __MDSPAN_OP(m, 1, 1, 1) = 42;
  auto sub0 = stdex::submdspan(m, 1, stdex::full_extent, stdex::full_extent);
  ASSERT_EQ(sub0.rank(),         2);
  ASSERT_EQ(sub0.rank_dynamic(), 0);
  ASSERT_EQ(sub0.extent(0),      3);
  ASSERT_EQ(sub0.extent(1),      4);
  ASSERT_EQ((__MDSPAN_OP(sub0, 1, 1)), 42);
  auto sub1 = stdex::submdspan(sub0, 1, stdex::full_extent);
  ASSERT_EQ(sub1.rank(),         1);
  ASSERT_EQ(sub1.rank_dynamic(), 0);
  ASSERT_EQ(sub1.extent(0),      4);
  ASSERT_EQ((__MDSPAN_OP(sub1,1)),42);
  auto sub2 = stdex::submdspan(sub1, 1);
  ASSERT_EQ(sub2.rank(),         0);
  ASSERT_EQ(sub2.rank_dynamic(), 0);
  ASSERT_EQ((__MDSPAN_OP0(sub2)), 42);
}

TEST(TestSubmdspanLayoutRightStaticSizedPairs, test_submdspan_layout_right_static_sized_pairs) {
  std::vector<int> d(2 * 3 * 4, 0);
  stdex::mdspan<int, stdex::extents<2, 3, 4>> m(d.data());
  __MDSPAN_OP(m, 1, 1, 1) = 42;
  auto sub0 = stdex::submdspan(m, std::pair<int,int>{1, 2}, std::pair<int,int>{1, 3}, std::pair<int,int>{1, 4});
  ASSERT_EQ(sub0.rank(),         3);
  ASSERT_EQ(sub0.rank_dynamic(), 3);
  ASSERT_EQ(sub0.extent(0),      1);
  ASSERT_EQ(sub0.extent(1),      2);
  ASSERT_EQ(sub0.extent(2),      3);
  ASSERT_EQ((__MDSPAN_OP(sub0, 0, 0, 0)), 42);
}

TEST(TestSubmdspanLayoutRightStaticSizedTuples, test_submdspan_layout_right_static_sized_tuples) {
  std::vector<int> d(2 * 3 * 4, 0);
  stdex::mdspan<int, stdex::extents<2, 3, 4>> m(d.data());
  __MDSPAN_OP(m, 1, 1, 1) = 42;
  auto sub0 = stdex::submdspan(m, std::tuple<int,int>{1, 2}, std::tuple<int,int>{1, 3}, std::tuple<int,int>{1, 4});
  ASSERT_EQ(sub0.rank(),         3);
  ASSERT_EQ(sub0.rank_dynamic(), 3);
  ASSERT_EQ(sub0.extent(0),      1);
  ASSERT_EQ(sub0.extent(1),      2);
  ASSERT_EQ(sub0.extent(2),      3);
  ASSERT_EQ((__MDSPAN_OP(sub0, 0, 0, 0)),       42);
}

template <typename Upstream> struct layout_diagonal {
public:
  template <class Extents> class mapping {
    stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent> upstream_;
    template <class> friend class mapping;

  public:
    using size_type = typename Extents::size_type;
    using layout_type = layout_diagonal;
    using extents_type = Extents;

  public:
    constexpr mapping() noexcept = default;
    constexpr mapping(mapping const &) noexcept = default;
    constexpr mapping(mapping &&) noexcept = default;
    constexpr mapping(
        stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent> const
            &that) noexcept
        : upstream_{that} {}

    constexpr mapping &operator=(mapping const &) noexcept = default;
    constexpr mapping &operator=(mapping &&) noexcept = default;
    ~mapping() noexcept = default;

    static constexpr bool is_always_strided() noexcept { return true; }
    static constexpr bool is_always_contiguous() noexcept { return false; }
    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_unique() noexcept { return true; }
    constexpr bool is_contiguous() const noexcept { return false; }
    constexpr bool is_strided() const noexcept { return false; }

    template <typename Index>
    constexpr size_type operator()(Index idx) const noexcept {
      using E = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
      E extent{upstream_.extent(0), upstream_.extent(0)};
      using M = typename Upstream::template mapping<E>;
      return M{extent}(idx, idx);
    }
  };
};

template <typename T, typename Extents, typename Upstream>
using diagonal_mdspan = stdex::mdspan<T, Extents, layout_diagonal<Upstream>,
                                      stdex::default_accessor<T>>;

template <typename T, typename Extents, typename Layout>
auto make_diagonal(stdex::mdspan<T, Extents, Layout> src) {
  // fixme: pass mapping type.
  typename layout_diagonal<Layout>::template mapping<
      stdex::extents<stdex::dynamic_extent>>
      m(src.extents());
  return diagonal_mdspan<T, stdex::extents<stdex::dynamic_extent>, Layout>{
      src.data(), m};
}

template <typename T, typename Extents, typename Layout, typename Accessor>
auto make_diagonal_1(stdex::mdspan<T, Extents, Layout, Accessor> const src) {
  static_assert(Extents::rank() == 2, "");
  auto mapping = src.mapping();
  auto s0 = mapping.stride(0);
  auto s1 = mapping.stride(1);
  auto e = mapping.extents();
  auto ret_stride = s0 + s1;
  auto ret_shape = std::max(e.extent(0), e.extent(1));
  stdex::extents<stdex::dynamic_extent> ret_e{ret_shape};
  std::array<size_t, 1> ret_strides{ret_stride};

  stdex::layout_stride::mapping<stdex::extents<stdex::dynamic_extent>>
      ret_mapping{ret_e, ret_strides};
  stdex::mdspan<T, decltype(ret_e), stdex::layout_stride, Accessor> ret{
      src.data(), ret_mapping};
  return ret;
}

template <typename T, bool is_device> struct accessor_base {
public:
  using offset_policy = accessor_base;
  using element_type = T;
  using reference = T &;
  using pointer = T *;

  constexpr accessor_base() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
      class OtherElementType,
      /* requires */ (_MDSPAN_TRAIT(
          std::is_convertible,
          typename accessor_base<OtherElementType, is_device>::element_type (
                  *)[],
          element_type (*)[])))
  MDSPAN_INLINE_FUNCTION
  constexpr accessor_base(accessor_base<OtherElementType, is_device>) noexcept {
  }

  MDSPAN_INLINE_FUNCTION
  constexpr pointer offset(pointer p, size_t i) const noexcept { return p + i; }

  MDSPAN_FORCE_INLINE_FUNCTION
  constexpr reference access(pointer p, size_t i) const noexcept {
    return p[i];
  }
};

template <typename T> using device_accessor = accessor_base<T, true>;
template <typename T> using host_accessor = accessor_base<T, false>;;

TEST(TestAccessor, host_device) {
  std::vector<int> d(3 * 3, 0);
  std::iota(d.begin(), d.end(), 0);
  stdex::mdspan<int,
                stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>,
                stdex::layout_right, device_accessor<int>>
      d_m(d.data(), 3, 3);
  stdex::mdspan<int,
                stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>,
                stdex::layout_right, device_accessor<int>>
      d_m_1(d_m);

  /**
   * stdex::mdspan<int,
   *               stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>,
   *               stdex::layout_right, host_accessor<int>>
   *     h_m(d_m);
   */
}

TEST(TestSubmdspan, test_diagonal) {
  std::vector<int> d(3 * 3, 0);
  std::iota(d.begin(), d.end(), 0);
  stdex::mdspan<int,
                stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>
      m(d.data(), 3, 3);

  auto mapping = m.mapping();
  auto accessor = m.accessor();

  auto diag = make_diagonal(m);
  stdex::mdspan<int const,
                stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>>
      c_m{m};
  ASSERT_EQ(diag(1), diag[1]);
  ASSERT_EQ(diag[1], 4);
  {
    auto diag = make_diagonal_1(c_m);
    ASSERT_EQ(diag(1), diag[1]);
    ASSERT_EQ(diag[1], 4);
    auto subspan = stdex::submdspan(diag, std::pair<size_t, size_t>{1, 3});
    for (size_t i = 0; i < subspan.extent(0); ++i) {
      std::cout << subspan(i) << std::endl;
    }
  }
}

/**
 * A layout for tridiagonal matrix that's compatible with cusolver.
 */
struct layout_tridiagonal {
  template <class Extents> class mapping {
  public:
    using size_type = typename Extents::size_type;
    using layout_type = layout_tridiagonal;
    using extents_type = Extents;

  private:
    extents_type extents_;
    template <typename Index>
    constexpr static size_t _offset(size_type n, Index i, Index j) {
      if (i == j) {
        return i;
      } else if (i < j) {
        return n + j - 1;
      } else {
        return n + n - 1 + i - 1;
      }
    }

  public:
    constexpr mapping() noexcept = default;
    constexpr mapping(extents_type e) noexcept : extents_{std::move(e)} {}
    constexpr mapping(mapping const &) noexcept = default;
    constexpr mapping(mapping &&) noexcept = default;
    constexpr mapping &operator=(mapping const &) noexcept = default;
    constexpr mapping &operator=(mapping &&) noexcept = default;
    ~mapping() noexcept = default;

    // fixme: verify this
    static constexpr bool is_always_strided() noexcept { return true; }
    static constexpr bool is_always_contiguous() noexcept { return false; }
    static constexpr bool is_always_unique() noexcept { return true; }
    static constexpr bool is_unique() noexcept { return true; }
    constexpr bool is_contiguous() const noexcept { return false; }
    constexpr bool is_strided() const noexcept { return true; }

    template <typename Index>
    constexpr size_type operator()(Index i, Index j) const noexcept {
      return _offset(extents_.extent(0), i, j);
    }

    constexpr size_type required_span_size() const noexcept {
      return this->extents_.extent(0) * this->extents_.extent(1);
    }
    constexpr Extents extents() const noexcept { return extents_; }
  };
};

template <typename T> auto make_tridiagonal(T *ptr, size_t n) {
  using extent_t = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
  using tridiag_matrix_t =
      stdex::mdspan<T, extent_t, layout_tridiagonal, host_accessor<T>>;
  auto e = extent_t{n, n};
  auto mapping = layout_tridiagonal::mapping<extent_t>(e);
  tridiag_matrix_t m{ptr, mapping};
  return m;
}

TEST(TestTridiagonal, Basic) {
  size_t n = 8;
  std::vector<int> d(n + (n - 1) * 2, 0);
  std::iota(d.begin(), d.end(), 0);
  auto tridiag = make_tridiagonal(d.data(), n);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      if (i == j || i + 1 == j || j + 1 == i) {
        std::cout << tridiag(i, j) << ", ";
      } else {
        std::cout << 0 << ", ";
      }
    }
    std::cout << std::endl;
  }
}

//template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>


using submdspan_test_types =
  ::testing::Types<
      // LayoutLeft to LayoutLeft
      std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<1>,stdex::dextents<1>, stdex::full_extent_t>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<1>,stdex::dextents<1>, std::pair<int,int>>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<1>,stdex::dextents<0>, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<2>,stdex::dextents<2>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<2>,stdex::dextents<2>, stdex::full_extent_t, std::pair<int,int>>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<2>,stdex::dextents<1>, stdex::full_extent_t, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<3>,stdex::dextents<3>, stdex::full_extent_t, stdex::full_extent_t, std::pair<int,int>>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<3>,stdex::dextents<2>, stdex::full_extent_t, std::pair<int,int>, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<3>,stdex::dextents<1>, stdex::full_extent_t, int, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<3>,stdex::dextents<1>, std::pair<int,int>, int, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<6>,stdex::dextents<3>, stdex::full_extent_t, stdex::full_extent_t, std::pair<int,int>, int, int, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<6>,stdex::dextents<2>, stdex::full_extent_t, std::pair<int,int>, int, int, int, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<6>,stdex::dextents<1>, stdex::full_extent_t, int, int, int ,int, int>
    , std::tuple<stdex::layout_left, stdex::layout_left, stdex::dextents<6>,stdex::dextents<1>, std::pair<int,int>, int, int, int, int, int>
    // LayoutRight to LayoutRight
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<1>,stdex::dextents<1>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<1>,stdex::dextents<1>, std::pair<int,int>>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<1>,stdex::dextents<0>, int>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<2>,stdex::dextents<2>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<2>,stdex::dextents<2>, std::pair<int,int>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<2>,stdex::dextents<1>, int, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<3>,stdex::dextents<3>, std::pair<int,int>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<3>,stdex::dextents<2>, int, std::pair<int,int>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<3>,stdex::dextents<1>, int, int, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<6>,stdex::dextents<3>, int, int, int, std::pair<int,int>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<6>,stdex::dextents<2>, int, int, int, int, std::pair<int,int>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::dextents<6>,stdex::dextents<1>, int, int, int, int, int, stdex::full_extent_t>
    // LayoutRight to LayoutRight Check Extents Preservation
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1>,stdex::extents<1>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1>,stdex::extents<dyn>, std::pair<int,int>>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1>,stdex::extents<>, int>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2>,stdex::extents<1,2>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2>,stdex::extents<dyn,2>, std::pair<int,int>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2>,stdex::extents<2>, int, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2,3>,stdex::extents<dyn,2,3>, std::pair<int,int>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2,3>,stdex::extents<dyn,3>, int, std::pair<int,int>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2,3>,stdex::extents<3>, int, int, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2,3,4,5,6>,stdex::extents<dyn,5,6>, int, int, int, std::pair<int,int>, stdex::full_extent_t, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2,3,4,5,6>,stdex::extents<dyn,6>, int, int, int, int, std::pair<int,int>, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_right, stdex::extents<1,2,3,4,5,6>,stdex::extents<6>, int, int, int, int, int, stdex::full_extent_t>

    , std::tuple<stdex::layout_right, stdex::layout_stride, stdex::extents<1,2,3,4,5,6>,stdex::extents<1,dyn,6>, stdex::full_extent_t, int, std::pair<int,int>, int, int, stdex::full_extent_t>
    , std::tuple<stdex::layout_right, stdex::layout_stride, stdex::extents<1,2,3,4,5,6>,stdex::extents<2,dyn,5>, int, stdex::full_extent_t, std::pair<int,int>, int, stdex::full_extent_t, int>
    >;

template<class T> struct TestSubMDSpan;

template<class LayoutOrg, class LayoutSub, class ExtentsOrg, class ExtentsSub, class ... SubArgs>
struct TestSubMDSpan<
  std::tuple<LayoutOrg,
             LayoutSub,
             ExtentsOrg,
             ExtentsSub,
             SubArgs...>>
  : public ::testing::Test {

  using mds_org_t = stdex::mdspan<int, ExtentsOrg, LayoutOrg>;
  using mds_sub_t = stdex::mdspan<int, ExtentsSub, LayoutSub>;

  using mds_sub_deduced_t = decltype(stdex::submdspan(mds_org_t(), SubArgs()...));
  using sub_args_t = std::tuple<SubArgs...>;

};


TYPED_TEST_SUITE(TestSubMDSpan, submdspan_test_types);

TYPED_TEST(TestSubMDSpan, submdspan_return_type) {
  static_assert(std::is_same<typename TestFixture::mds_sub_t,
                             typename TestFixture::mds_sub_deduced_t>::value,
                "SubMDSpan: wrong return type");

}
