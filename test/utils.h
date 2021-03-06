#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <Eigen/Core>
#include <gtest/gtest.h>

template<typename T, int rows, int cols>
testing::AssertionResult MatrixMatch(const Eigen::Matrix<T, rows, cols> &expected, const Eigen::Matrix<T, rows, cols> &actual, T eps)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
                        T e = expected(i, j), a = actual(i, j);
                        if (abs(e - a) >= eps) {
                                return ::testing::AssertionFailure() << "expected[" << i << ", " << j << "] (" << e << ") != actual[" << i << ", " << j << "] (" << a << ")";
                        }
                }
        return ::testing::AssertionSuccess();
}

template<typename T>
testing::AssertionResult vectorMatch(const std::vector<T> &expected, const std::vector<T> &actual)
{
    if (expected.size() != actual.size())
        return ::testing::AssertionFailure() << "expected.size (" << expected.size() << ") != actual.size (" << actual.size() << ")";
    for (size_t i = 0; i < expected.size(); i++) {
        T e = expected[i], a = actual[i];
        if (e != a) {
            return ::testing::AssertionFailure() << "expected[" << i << "] (" << e << ") != actual[" << i << "] (" << a << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

template<typename T>
testing::AssertionResult vectorMatch(const std::vector<T> &actual, const T &expected)
{
    for (size_t i = 0; i < actual.size(); i++) {
        T a = actual[i];
        if (a != expected) {
            return ::testing::AssertionFailure() << "expected (" << expected << ") != actual[" << i << "] (" << a << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

#endif // UTILS_H
