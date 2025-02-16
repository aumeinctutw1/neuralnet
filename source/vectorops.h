#ifndef VECTOROPS_H
#define VECTOROPS_H

/*
*   Vector operations bases on std::vector<T>
*   Matrix operations based on std::vector<std::vector<T>>
*/

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <functional>

template <typename T>
void uniform_random_initialization (
    std::vector<std::vector<T>> &A,
    const std::pair<size_t, size_t> &shape,
    const T &low, const T &high
){
    A.clear();  
    /* Uniform distribution in range [low, high] */
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_real_distribution<T> distribution(low, high);
    for (size_t i = 0; i < shape.first; i++) {  
        std::vector<T> row;  
        row.resize(shape.second);
        for (auto &r : row) {             
            r = distribution(generator);  
            if (r > 1.0) {
                std::cerr << "Random number greater than 1.0: " << r << std::endl;
            }
        }
        A.push_back(row);  
    }
    return;
}

template <typename T>
void unit_matrix_initialization (
    std::vector<std::vector<T>> &A,
    const std::pair<size_t, size_t> &shape
){
    A.clear();  
    for (size_t i = 0; i < shape.first; i++) {  
        std::vector<T> row;  
        row.resize(shape.second);
        row[i] = T(1);
        A.push_back(row);  
    }
    return;
}

template <typename T>
std::vector<T> matrix_vector_multiplication (
    const std::vector<std::vector<T>> &A,
    const std::vector<T> &B
){
    try {
        /* check dimensions */
        if (A.empty() || A.at(0).size() != B.size()) {
            throw std::invalid_argument("Matrix and vector dimensions do not match");
        }

        std::vector<T> C;
        for (size_t i = 0; i < A.size(); i++) {
            T sum = 0;
            for (size_t j = 0; j < B.size(); j++) {
                sum += A.at(i).at(j) * B.at(j);
            }
            C.push_back(sum);
        }

        return C;
    } catch (const std::exception &e) {
        std::cerr << "Exception in matrix vector multiplication: " <<  e.what() << std::endl;
        throw;
    }
}

template <typename T>
std::vector<std::vector<T>> transpose_matrix (
    const std::vector<std::vector<T>> &A
){
    std::vector<std::vector<T>> B;
    for (size_t i = 0; i < A.at(0).size(); i++) {
        std::vector<T> row;
        for (size_t j = 0; j < A.size(); j++) {
            row.push_back(A.at(j).at(i));
        }
        B.push_back(row);
    }
    return B;
}

template <typename T>
std::vector<std::vector<T>> scalar_matrix_multiplication (
    const T &scalar,
    const std::vector<std::vector<T>> &A
){
    std::vector<std::vector<T>> B;
    for (size_t i = 0; i < A.size(); i++) {
        std::vector<T> row;
        for (size_t j = 0; j < A.at(0).size(); j++) {
            row.push_back(A.at(i).at(j) * scalar);
        }
        B.push_back(row);
    }
    return B;
}

template <typename T>
std::vector<std::vector<T>> matrix_matrix_addition (
    const std::vector<std::vector<T>> &A,
    const std::vector<std::vector<T>> &B
){
    std::vector<std::vector<T>> C;

    if (A.size() != B.size() || A.at(0).size() != B.at(0).size()) {
        throw std::invalid_argument("Matrix dimensions for addition do not match");
    }

    try {
        for (size_t i = 0; i < A.size(); i++) {
            std::vector<T> row;
            for (size_t j = 0; j < A.at(0).size(); j++) {
                row.push_back(A.at(i).at(j) + B.at(i).at(j));
            }
            C.push_back(row);
        }
        return C;
    } catch (const std::exception &e) {
        std::cerr << "Exception in adding matrices: " <<  e.what() << std::endl;
        throw;
    }
}

template <typename T>
std::vector<T> subtract_vectors(std::vector<T> A, std::vector<T> B) {
    std::vector<T> C;

    if (A.size() != B.size()) {
        throw std::invalid_argument("Vector dimensions do not match");
    }

    try {
        for (size_t i = 0; i < A.size(); i++) {
            C.push_back(A.at(i) - B.at(i));
        }
        return C;
    } catch (const std::exception &e) {
        std::cerr << "Exception in subtracting vectors: " <<  e.what() << std::endl;
        throw;
    }
}

template <typename T>
void apply_function (
    std::vector<T> &A,
    const std::function<T(T)> &func
){
    for (auto &a : A) {
        a = func(a);
    }
    return;
}

template <typename T>
void print_vector (
    const std::vector<T> &A
){
    for (auto &a : A) {
        std::cout << a << " ";
    }
    std::cout << std::endl;
    return;
}

template <typename T>
void print_matrix(
    const std::vector<std::vector<T>> &A
){
    for (auto &row : A) {
        for (auto &r : row) {
            std::cout << r << " ";
        }
        std::cout << std::endl;
    }
    return;
}

#endif