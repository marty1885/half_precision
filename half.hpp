#pragma once

#include "fp16.h"
#include <cstdint>
#include <type_traits>
#include <numeric>
#include <functional>

namespace half_precision
{

struct half
{

	half() = default;
	half(float v) : storage_(fp16_ieee_from_fp32_value(v)) {}

	operator float() const
	{
		return fp16_ieee_to_fp32_value(storage_);
	}

	uint16_t storage_;
};

template <typename T>
struct half_operation_result
{
	using type = std::conditional_t<std::is_same_v<T, half>, half
		, std::conditional_t<std::numeric_limits<T>::is_integer, half, T>>;
};

template <typename T>
using half_operation_result_t = typename half_operation_result<T>::type;

template <typename T>
auto convert_native_type(T v)
{
	using type = std::conditional_t<std::is_same_v<T, half>, float, T>;
	type res;
	if constexpr(std::is_same_v<T, half>)
		res = fp16_ieee_to_fp32_value(v.storage_);
	else
		res = v;
	return res;
}

template <typename T1, typename T2, typename Op>
auto generic_half_operation(T1 self, T2 other, Op op)
{
	//Make sure one of the parameters is a half
	static_assert(std::is_same_v<T1, half> || std::is_same_v<T2, half>);

	auto rhs = convert_native_type(self);
	auto lhs = convert_native_type(other);

	using the_other_type = std::conditional_t<std::is_same_v<T1, half>, T2, T1>;

	return half_operation_result_t<the_other_type>(op(rhs, lhs));
}

// Uniary operators
inline half operator -(half self)
{
	return half(-1.f*self);
}

inline half operator +(half self)
{
	return self;
}

// Binary operations

inline auto operator +(half self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a+b;});
}

inline auto operator -(half self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a-b;});
}

inline auto operator *(half self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a*b;});
}

inline auto operator /(half self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a/b;});
}

template <typename T>
inline auto operator +(half self, T other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a+b;});
}

template <typename T>
inline auto operator -(half self, T other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a-b;});
}

template <typename T>
inline auto operator *(half self, T other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a*b;});
}

template <typename T>
inline auto operator /(half self, T other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a/b;});
}

template <typename T>
inline auto operator +(T self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a+b;});
}

template <typename T>
inline auto operator -(T self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a-b;});
}

template <typename T>
inline auto operator *(T self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a*b;});
}

template <typename T>
inline auto operator /(T self, half other)
{
	return generic_half_operation(self, other, [](auto a, auto b){return a/b;});
}

// Base comparsion operators

inline bool operator <(half self, half other)
{
	return (float)self < (float)other;
}

inline bool operator >(half self, half other)
{
	return (float)self > (float)other;
}

inline bool operator ==(half self, half other)
{
	return (float)self == (float)other;
}

template <typename T>
inline bool operator <(half self, T other)
{
	return (float)self < other;
}

template <typename T>
inline bool operator >(half self, T other)
{
	return (float)self > other;
}

template <typename T>
inline bool operator ==(half self, T other)
{
	return (float)self == other;
}

template <typename T>
inline bool operator <(T self, half other)
{
	return self < (float)other;
}

template <typename T>
inline bool operator >(T self, half other)
{
	return self > (float)other;
}

template <typename T>
inline bool operator ==(T self, half other)
{
	return self == (float)other;
}

// Composed campersion operators

inline bool operator !=(half self, half other)
{
	return !(self == other);
}

inline bool operator >=(half self, half other)
{
	return self == other || self > other;
}

inline bool operator <=(half self, half other)
{
	return self == other || self < other;
}

template <typename T>
inline bool operator !=(T self, half other)
{
	return !(self == other);
}

template <typename T>
inline bool operator >=(T self, half other)
{
	return self == other || self > other;
}

template <typename T>
inline bool operator <=(T self, half other)
{
	return self == other || self < other;
}

template <typename T>
inline bool operator !=(half self, T other)
{
	return !(self == other);
}

template <typename T>
inline bool operator >=(half self, T other)
{
	return self == other || self > other;
}

template <typename T>
inline bool operator <=(half self, T other)
{
	return self == other || self < other;
}

}