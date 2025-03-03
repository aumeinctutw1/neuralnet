#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

/* NN Stuff */
#include "neuralnetwork.h"
#include "activations.h"
#include "vectorops.h"

/* scales integer input to doubles between 0.01 and 1.0 */
double scaleData(int input) {
    double scaled_input = (input / 255.0 * 0.98) + 0.01;
    return scaled_input;
}

template <typename T>
std::vector<std::vector<T>> readCSV(std::string filepath) {
    std::vector<std::vector<T>> data;

    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("csv file not found: " + filepath);
    }

    std::ifstream infile{filepath};
    std::string line;

    while (std::getline(infile, line)) {
        std::vector<T> csv_line;
        std::istringstream iss{line};

        int j = 0;
        /* read every value in the line */
        for (int i; iss >> i;) {
            /* dont scale the first value */
            if (j == 0) {
                csv_line.push_back(i);
            } else {
                csv_line.push_back(scaleData(i));
            }

            if (iss.peek() == ',') {
                iss.ignore();
            }
            ++j;
        }
        data.push_back(csv_line);
    }

    return data;
}

/*
*   Specific to the mnist training data
*/
template <typename T>
std::vector<T> getInput(const std::vector<T> &training_data) {
    std::vector<T> input;
    try {
        auto it = std::next(training_data.begin(), 1);
        input.insert(input.end(), it, training_data.end());
    } catch (const std::out_of_range &err) {
        std::cout << "Out of range err: " << err.what() << std::endl;
        throw std::runtime_error("Err get input");
    } 

    return input;
}

/*
*   Specific to the mnist training data
*/
template <typename T>
std::vector<T> getTargets(const std::vector<T> &training_data, int onodes) {
    /* set every target intially to 0.01 */
    std::vector<T> targets(onodes, 0.01);

    /* set the specified target from the training data to 0.99  */
    try {
        targets.at(training_data.at(0)) = 0.99;
    } catch (const std::out_of_range &err) {
        std::cout << "Out of Range: " << err.what() << " tried accessing training data: " << training_data.at(0) << std::endl;
        throw std::runtime_error("Err");
    }
    
    return targets;
}

template <typename T>
int getIndexOfTarget(const std::vector<T> &output) {
    auto it = std::max_element(output.begin(), output.end());   
    return std::distance(output.begin(), it);
}

template <typename T>
void testModel(std::string test_csv, NeuralNetwork<T> &nn) {
    /* query the model with test data */
    std::cout << "Querying model with test data" << std::endl;
    std::vector<std::vector<float>> test_data = readCSV<float>(test_csv);
    int scoreboard = 0; 
    for (int i = 0; i < test_data.size(); i++) {
        std::vector<float> input = getInput<float>(test_data.at(i));
        std::vector<float> prediction = nn.query(input);
        std::cout << "Prediction: " << getIndexOfTarget<float>(prediction) << " ";
        std::cout << "Target: " << test_data.at(i).at(0) << std::endl;
        if (getIndexOfTarget<float>(prediction) == test_data.at(i).at(0)) {
            scoreboard++;
        }
    }

    /* print the accuracy */
    std::cout << "Accuracy: " << (scoreboard / (float)test_data.size()) * 100 << "%" << std::endl;
}

template <typename T>
void trainModel(std::string training_csv, NeuralNetwork<T> &nn) {
    /* read training csv */
    std::vector<std::vector<float>> training_data = readCSV<float>(training_csv);
    
    /* train the model */
    for (int i = 0; i < training_data.size(); i++) {
        std::vector<float> input = getInput<float>(training_data.at(i));
        std::vector<float> target = getTargets<float>(training_data.at(i), 10);
        nn.train(input, target);
    }
}

template <typename T>
void queryModel(NeuralNetwork<T> &nn, std::vector<T> input) {
    std::cout << "Querying model with input" << std::endl;
    std::vector<T> output = nn.query(input);
    std::cout << "Prediction: " << getIndexOfTarget<T>(output) << std::endl;
}

int main (int argc, const char *argv[]) {
    try {
        /* create a neural network */
        NeuralNetwork nn = NeuralNetwork<float>({{784, "none"}, {100, "sigmoid"}, {10, "sigmoid"}}, 0.3);
        if (std::filesystem::exists("model.txt")) {
            std::cout << "Loading model from file" << std::endl;
            nn.loadModel("model.txt");
        }
    } catch (const std::exception &e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}