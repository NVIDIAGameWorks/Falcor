//
// Copyright (c) 2020-2020 Martin Moene
//
// https://github.com/martinmoene/bit-lite
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Falcor: Stripped down version assuming C++17 and placing into namespace fstd.

#pragma once

#include <cstring>      // std::memcpy()
#include <climits>      // CHAR_BIT
#include <limits>       // std::numeric_limits<>
#include <type_traits>

#if 0

// 26.5.3, bit_cast

template< class To, class From > constexpr To bit_cast( From const & from ) noexcept;

// 26.5.4, integral powers of 2

template< class T > constexpr bool has_single_bit(T x) noexcept;
template< class T > constexpr T bit_ceil(T x);
template< class T > constexpr T bit_floor(T x) noexcept;
template< class T > constexpr T bit_width(T x) noexcept;

// 26.5.5, rotating

template< class T > [[nodiscard]] constexpr T rotl(T x, int s) noexcept;
template< class T > [[nodiscard]] constexpr T rotr(T x, int s) noexcept;

// 26.5.6, counting

template< class T > constexpr int countl_zero(T x) noexcept;
template< class T > constexpr int countl_one(T x) noexcept;
template< class T > constexpr int countr_zero(T x) noexcept;
template< class T > constexpr int countr_one(T x) noexcept;
template< class T > constexpr int popcount(T x) noexcept;

#endif // 0: For reference

namespace fstd {
namespace bit {

template< typename T >
constexpr T bitmask( int i )
{
    return static_cast<T>( T{1} << i );
}

// C++20 emulation:

namespace std20 {

template< class T, class U >
struct same_as : std::integral_constant<bool, std::is_same<T,U>::value && std::is_same<U,T>::value> {};

} // namepsace std20

//
// Implementation:
//

// 26.5.3, bit_cast

// constexpr support needs compiler magic

template< class To, class From >
/*constexpr*/
typename std::enable_if<
    ( (sizeof(To) == sizeof(From))
        && std::is_trivially_copyable<From>::value
        && std::is_trivial<To>::value
        && (std::is_copy_constructible<To>::value || std::is_move_constructible<To>::value)
    ) , To>::type
bit_cast(From const & src) noexcept
{
    To dst;
    std::memcpy( &dst, &src, sizeof(To) );
    return dst;
}

// 26.5.5, rotating

template< class T >
[[nodiscard]] constexpr T rotr_impl(T x, int s) noexcept;

template< class T >
[[nodiscard]] constexpr T rotl_impl(T x, int s) noexcept
{
    constexpr int N = std::numeric_limits<T>::digits;
    const int r = s % N;

    if ( r == 0 )
        return x;
    else if ( r > 0 )
        return static_cast<T>( (x << r) | (x >> (N - r)) );
    else /*if ( r < 0 )*/
        return rotr_impl( x, -r );
}

template< class T >
[[nodiscard]] constexpr T rotr_impl(T x, int s) noexcept
{
    constexpr int N = std::numeric_limits<T>::digits;
    const int r = s % N;

    if ( r == 0 )
        return x;
    else if ( r > 0 )
        return static_cast<T>( (x >> r) | (x << (N - r)) );
    else /*if ( r < 0 )*/
        return rotl_impl( x, -r );
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
[[nodiscard]] constexpr T rotl(T x, int s) noexcept
{
    return rotl_impl( x, s );
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
[[nodiscard]] constexpr T rotr(T x, int s) noexcept
{
    return rotr_impl( x, s );
}

// 26.5.6, counting

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr int countl_zero(T x) noexcept
{
    constexpr int N1 = CHAR_BIT * sizeof(T) - 1;

    int result = 0;
    for( int i = N1; i >= 0; --i, ++result )
    {
        if ( 0 != (x & bitmask<T>(i)) )
            break;
    }
    return result;
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr int countl_one(T x) noexcept
{
    constexpr int N1 = CHAR_BIT * sizeof(T) - 1;

    int result = 0;
    for( int i = N1; i >= 0; --i, ++result )
    {
        if ( 0 == (x & bitmask<T>(i)) )
            break;
    }
    return result;
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr int countr_zero(T x) noexcept
{
    constexpr int N = CHAR_BIT * sizeof(T);

    int result = 0;
    for( int i = 0; i < N; ++i, ++result )
    {
        if ( 0 != (x & bitmask<T>(i)) )
            break;
    }
    return result;
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr int countr_one(T x) noexcept
{
    constexpr int N = CHAR_BIT * sizeof(T);

    int result = 0;
    for( int i = 0; i < N; ++i, ++result )
    {
        if ( 0 == (x & bitmask<T>(i)) )
            break;
    }
    return result;
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr int popcount(T x) noexcept
{
    constexpr int N = CHAR_BIT * sizeof(T);

    int result = 0;
    for( int i = 0; i < N; ++i )
    {
        if ( 0 != (x & bitmask<T>(i)) )
            ++result;
    }
    return result;
}

// 26.5.4, integral powers of 2

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr bool has_single_bit(T x) noexcept
{
    return x != 0 && ( x & (x - 1) ) == 0;
    // return std::popcount(x) == 1;
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr T bit_width(T x) noexcept
{
    return static_cast<T>(std::numeric_limits<T>::digits - countl_zero(x) );
}

template< class T >
constexpr T bit_ceil_impl( T x, std::true_type /*case: same type*/)
{
    return T{1} << bit_width( T{x - 1} );
}

template< class T >
constexpr T bit_ceil_impl( T x, std::false_type /*case: integral promotion*/ )
{
    constexpr T offset_for_ub =
        static_cast<T>( std::numeric_limits<unsigned>::digits - std::numeric_limits<T>::digits );

    return T{ 1u << ( bit_width(T{x - 1}) + offset_for_ub ) >> offset_for_ub };
}

// ToDo: pre-C++11 behaviour for types subject to integral promotion.

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr T bit_ceil(T x)
{
    return ( x <= 1u ) ? T{1} : bit_ceil_impl( x, std20::same_as<T, decltype(+x)>{} );
}

template< class T, typename std::enable_if<std::is_unsigned<T>::value, int >::type = 0 >
constexpr T bit_floor(T x) noexcept
{
    return (x != 0) ? T{1} << (bit_width(x) - 1) : 0;
}

// 26.5.7, endian

enum class endian
{
#ifdef _WIN32
    little = 0,
    big    = 1,
    native = little
#else
    little = __ORDER_LITTLE_ENDIAN__,
    big    = __ORDER_BIG_ENDIAN__,
    native = __BYTE_ORDER__
#endif
};

} // namespace bit
} // namespace fstd

//
// Make type available in namespace fstd:
//

namespace fstd
{
    using bit::bit_cast;

    using bit::has_single_bit;
    using bit::bit_ceil;
    using bit::bit_floor;
    using bit::bit_width;

    using bit::rotl;
    using bit::rotr;

    using bit::countl_zero;
    using bit::countl_one;
    using bit::countr_zero;
    using bit::countr_one;
    using bit::popcount;

    using bit::endian;
}
