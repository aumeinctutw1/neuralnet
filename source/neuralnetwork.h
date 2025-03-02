#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <string>

#include "layer.h"
#include "vectorops.h"

template <typename T>
class NeuralNetwork {
    public:
        NeuralNetwork(const std::vector<std::pair<int, std::string>> &shape, float learningRate);
        ~NeuralNetwork();

        void train(std::vector<T> input, std::vector<T> target);
        std::vector<T> query(std::vector<T> input);
        void saveModel(const std::string &path);
        void loadModel(const std::string &path);
        void printweights(); 

    private:
        std::vector<Layer<T>> m_layers;
        float m_learningRate;
};

template <typename T>
NeuralNetwork<T>::NeuralNetwork(const std::vector<std::pair<int, std::string>> &shape, float learningRate):
    m_learningRate(learningRate)
{
    /* first layer has no activation */
    if (shape[0].second != "none") {
        std::cerr << "First layer must have no activation" << std::endl;
        throw std::invalid_argument("First layer must have no activation");
    }

    /* atleast two layers are needed */
    if (shape.size() < 2) {
        std::cerr << "Atleast two layers are needed" << std::endl;
        throw std::invalid_argument("Atleast two layers are needed");
    }

    /* init input layer */
    m_layers.push_back(Layer<T>(
        shape[0].first,
        shape[0].second,
        {shape[0].first, shape[0].first},
        false
    ));

    /* init rest of the network */
    for (size_t i = 1; i < shape.size(); i++) {
        m_layers.push_back(Layer<T>(
            shape[i].first,
            shape[i].second,
            {shape[i].first, shape[i - 1].first},
            true
        ));
    }
}

template <typename T>
void NeuralNetwork<T>::loadModel(const std::string &path) {
    /* load model from filesystem */
    std::ifstream modelFile(path); 
    if (!modelFile.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        throw std::runtime_error("Could not open file");
    }

    /* clear exisiting layers */
    m_layers.clear();

    /* read learning rate */
    modelFile >> m_learningRate;

    /* read number of layers */
    int numLayers;
    modelFile >> numLayers;

    /* init the input layer */
    int inputNeurons;
    std::string inputActivation;
    modelFile >> inputNeurons >> inputActivation;
    m_layers.push_back(Layer<T>(
        inputNeurons,
        inputActivation,
        {inputNeurons, inputNeurons},
        false
    )); 

    /* create the hidden and output layers */
    for (int i = 1; i < numLayers + 1; i++) {
        int neurons;
        std::string activation;
        modelFile >> neurons >> activation;
        m_layers.push_back(Layer<T>(
            neurons,
            activation,
            {neurons, m_layers.at(i - 1).getNeurons()},
            true
        ));
    }

    /* read and set the weights for the hidden and output layer */
    for (int i = 1; i < m_layers.size(); i++) {
        std::vector<std::vector<T>> weights;
        for (int j = 0; j < m_layers.at(i).getNeurons(); j++) {
            std::vector<T> row;
            for (int k = 0; k < m_layers.at(i - 1).getNeurons(); k++) {
                T weight;
                modelFile >> weight;
                row.push_back(weight);
            }
            weights.push_back(row);
        }
        m_layers.at(i).setWeights(weights);
    }
}

template <typename T>
void NeuralNetwork<T>::saveModel(const std::string &path) {
    std::ofstream modelFile(path);
    if (!modelFile.is_open()) {
        std::cerr << "Could not open file: " << path << std::endl;
        throw std::runtime_error("Could not open file");
    }

    /* first line is the learningrate */
    modelFile << m_learningRate << std::endl;

    /* 
    *   store number of layers and there activation function in the txt file 
    */
    modelFile << m_layers.size() - 1 << std::endl;
    for (auto &layer : m_layers) {
        modelFile << layer.getNeurons() << " " << layer.getActivation() << std::endl;
    }
    modelFile << std::endl;

    /* store weights */
    for (auto &layer : m_layers) {
        /* skip the input layer */
        if (layer.getActivation() == "none") {
            continue;
        }

        for (auto &row : layer.getWeights()) {
            for (auto &col : row) {
                modelFile << col << " ";
            }
            modelFile << std::endl;
        }
        /* empty line as break between layers */
        modelFile << std::endl;
    }

    modelFile.close();
}

template<typename T>
NeuralNetwork<T>::~NeuralNetwork() {}

template<typename T>
void NeuralNetwork<T>::printweights() {
    for (auto &layer : m_layers) {
        print_matrix(layer.getWeights());
        std::cout << std::endl;
    }
}

template<typename T>
std::vector<T> NeuralNetwork<T>::query(std::vector<T> input) {
    /* check if input fits */
    if (input.size() != m_layers.at(0).getNeurons()) {
        std::cerr << "Input size does not match input layer size" << std::endl;
        throw std::invalid_argument("Input size does not match input layer size");
    }

    std::vector<T> output = input;

    /* forward pass */
    for(auto &layer : m_layers) {
        output = matrix_vector_multiplication(layer.getWeights(), output);
        apply_function(output, layer.getActivationFunction());
    }

    return output; 
}

template <typename T>   
void NeuralNetwork<T>::train(std::vector<T> input, std::vector<T> target) {
    /* check if input fits */
    if (input.size() != m_layers.at(0).getNeurons()) {
        std::cerr << "Input size does not match input layer size" << std::endl;
        throw std::invalid_argument("Input size does not match input layer size");
    }

    /* check if target fits */
    if (target.size() != m_layers.at(m_layers.size() - 1).getNeurons()) {
        std::cerr << "Target size does not match output layer size" << std::endl;
        throw std::invalid_argument("Target size does not match output layer size");
    }

    /* forward pass */
    std::vector<std::vector<T>> outputs;
    std::vector<T> output = input;

    for(int i = 1; i < m_layers.size(); i++) {
        /* multiply the input with the weights, then apply the activation function */
        output = matrix_vector_multiplication(m_layers.at(i).getWeights(), output);
        apply_function(output, m_layers.at(i).getActivationFunction());
        outputs.push_back(output);
    }

    /* backward pass */
    std::vector<std::vector<T>> errors;

    /* final error is simple subtraction of target - actual */
    std::vector<T> error = subtract_vectors(target, outputs.at(outputs.size() - 1));
    errors.push_back(error);

    for (int i = m_layers.size() - 2; i > 0; i--) {
        /* hidden errors are split by weights and recombined into hidden nodes */
        std::vector<T> nextError = matrix_vector_multiplication(transpose_matrix(m_layers.at(i + 1).getWeights()), error);
        error = nextError;
        errors.push_back(error);
    }

    /* reverse errors */
    std::reverse(errors.begin(), errors.end());

    /* check if outputs fit errors */
    if (outputs.size() != errors.size()) {
        std::cerr << "Output size does not match error size" << std::endl;
        throw std::invalid_argument("Output size does not match error size");
    }

    /* 
    *   update weights, start at final layer 
    *   to update the weights between the input and first hidden layer, the input is used
    */
    for (int i = 1; i < m_layers.size(); i++) {
        m_layers.at(i).updateWeights(errors.at(i - 1), outputs.at(i - 1), i == 1 ? input : outputs.at(i - 2), m_learningRate);
    }

    return;
}

#endif