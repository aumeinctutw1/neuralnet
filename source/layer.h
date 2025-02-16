#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>
#include <functional>

#include "vectorops.h"
#include "activations.h"

/*
* Layer class
* Stores a vector of neurons aka weights
*/
template <typename T>
class Layer {
    public:
        Layer(const int numNeurons, const std::string activationFunction, const std::pair<int, int> shape, const bool randomInit);
        ~Layer();

        int getNeurons() const { return m_neurons; }
        std::string getActivation() const { return m_activation; }
        std::vector<std::vector<T>> getWeights() const { return m_weights; }
        std::function<T(T)> getActivationFunction() const { return m_activationFunction; }
        void setWeights(const std::vector<std::vector<T>> &weights) { m_weights = weights; }
        void updateWeights(const std::vector<T> &error, const std::vector<T> &output, const std::vector<T> &prevOutput, const T learningRate);

    private:
        int m_neurons;
        std::string m_activation;
        std::vector<std::vector<T>> m_weights;
        std::function<T(T)> m_activationFunction;
};

template <typename T>
Layer<T>::Layer(const int numNeurons, const std::string activationFunction, const std::pair<int, int> shape, const bool randomInit):
    m_neurons(numNeurons),  m_activation(activationFunction)
{
    /* init weights */
    if (randomInit) {
        /* 
        *   If the low and high bound is to high, the sigmoid function will always return 1
        *   This is because the sigmoid function is limited to 0 and 1, thus the network will not learn
        *   0.5 seems to be a good starting bound
        */
        uniform_random_initialization<T>(m_weights, shape, -0.5, 0.5);
    } else {
        unit_matrix_initialization<T>(m_weights, shape);
    }

    /* init activation function */
    if (activationFunction == "sigmoid") {
        m_activationFunction = activations::sigmoid<T>;
    } else if (activationFunction == "relu") {
        m_activationFunction = activations::relu<T>;
    } else if (activationFunction == "tanh") {
        m_activationFunction = activations::tanh<T>;
    } else if (activationFunction == "none") {
        m_activationFunction = [](T x) { return x; };
    } else {
        std::cerr << "Invalid activation function" << std::endl;
        throw std::invalid_argument("Invalid activation function");
    }
}

template<typename T>
Layer<T>::~Layer() {}

template<typename T>
void Layer<T>::updateWeights(const std::vector<T> &error, const std::vector<T> &output, const std::vector<T> &prevOutput, const T learningRate) {
    /* check dimensions */
    if (error.size() != m_neurons || output.size() != m_neurons || prevOutput.size() != m_weights[0].size()) {
        throw std::invalid_argument("Dimensions dont fit to update the weights");
    }

    /* 
    *   deltaW(j,k) = lr * error(k) * output(k) * (1 - output(k)) * output(j)
    *   start at the output layer and move backwards
    *   It contains the errors, which can be added to the weights, in order to update the weights
    *
    *  k= rows, j = columns 
    */
    for (int k = 0; k < m_neurons; ++k) {
        for (int j = 0; j < m_weights[0].size(); ++j) {
            /* calculate the need change in the weight */
            T change_w = error[k] * output[k] * (1.0 - (output[k])) * prevOutput[j];
            /* calculate the new weight */
            m_weights[k][j] = m_weights[k][j] + (learningRate * change_w);
        }
    }
}

#endif