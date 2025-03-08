#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>

/* GUI Stuff */
#include <SFML/Graphics.hpp>

/* NN Stuff */
#include "neuralnetwork.h"
#include "activations.h"
#include "vectorops.h"

constexpr int CANVAS_WIDTH = 400;  // Pixels
constexpr int CANVAS_HEIGHT = 400; // Pixels
constexpr int CELL_SIZE = 4;       // Size of each "pixel" in the grid
constexpr int GRID_WIDTH = CANVAS_WIDTH / CELL_SIZE;   // 100 cells
constexpr int GRID_HEIGHT = CANVAS_HEIGHT / CELL_SIZE; // 100 cells

/* scales integer input to doubles between 0.01 and 1.0 */
template <typename T>
T scaleData(int input) {
    T scaled_input = (input / 255.0 * 0.98) + 0.01;
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
                csv_line.push_back(scaleData<T>(i));
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
*   Skip the first csv row entry (label), 
*   get only input vector
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
*   returns a vector, in which the index of the target (label)
*   has a value of 0.99, the other values in the vector are set to 0.01
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
    std::cout << "Querying model with input: " << std::endl;
    print_vector(input);
    std::vector<T> output = nn.query(input);
    std::cout << "Prediction: " << getIndexOfTarget<T>(output) << std::endl;
}

void saveToPNG(std::vector<int> mnistGrid, std::string filename) {
    sf::Image image({20, 20}, sf::Color::Black);

    for (u_int y = 0; y < 28; y++) {
        for (u_int x = 0; x < 28; x++) {
            int intensity = 255 - mnistGrid[y + 28 * x]; // Invert: 0 = black, 255 = white (MNIST style)
            image.setPixel({x, y}, sf::Color(intensity, intensity, intensity)); 
        }
    }

    if (image.saveToFile(filename)) {
        std::cout << "Saved to " << filename << "\n";
    } else {
        std::cerr << "Failed to save " << filename << "\n";
    }
}

/* to use the numbers drawn on the canvas, it needs to be down sampled to a 28x28 image */
std::vector<float> downSampleMNIST(std::vector<std::vector<int>> &grid) {
    const int blockSize = 4;
    constexpr int MNIST_SIZE = 28;
    std::vector<float> mnistGrid((MNIST_SIZE * MNIST_SIZE), 0.0);

    for (int y = 0; y < MNIST_SIZE; y++) {
        for (int x = 0; x < MNIST_SIZE; x++) {
            /* define the block cells to average 4x4 */
            int startX = x * 100 / MNIST_SIZE;
            int endX = (x + 1) * 100 / MNIST_SIZE;
            int startY = y * 100 / MNIST_SIZE;
            int endY = (y + 1) * 100 / MNIST_SIZE;

            int sum = 0;
            int count = 0;
            for (int gy = startY; gy < endY; gy++) {
                for (int gx = startX; gx < endX; gx++) {
                    sum += grid[gy][gx];
                    count++;
                }
            }

            float avg = static_cast<float>(sum) / count; // 0.0 - 1.0
            mnistGrid[y * 28 + x] = avg * 255; 
        }
    }
    return mnistGrid;
}

int main (int argc, const char *argv[]) {
    try {
        /* NN Stuff */
        NeuralNetwork nn = NeuralNetwork<float>({{784, "none"}, {100, "sigmoid"}, {10, "sigmoid"}}, 0.3);
        if (std::filesystem::exists("model.txt")) {
            std::cout << "Loading model from file" << std::endl;
            nn.loadModel("model.txt");

            std::vector<std::vector<float>> test_data = readCSV<float>("./mnist_data/mnist_test_10.csv");
            for (int i = 0; i < test_data.size(); i++) {
                std::vector<float> input = getInput<float>(test_data.at(i));
                std::vector<float> prediction = nn.query(input);
                std::cout << "Prediction: " << getIndexOfTarget<float>(prediction) << " ";
                std::cout << "Target: " << test_data.at(i).at(0) << std::endl;
            }
        }

        /* GUI Stuff */
        sf::RenderWindow window(sf::VideoMode({CANVAS_WIDTH, CANVAS_HEIGHT}), "Neuralnet");

        std::vector<std::vector<int>> grid(GRID_HEIGHT, std::vector<int>(GRID_WIDTH, 0));

        while (window.isOpen()) {
            while (const std::optional event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>()) {
                    window.close();
                }  
                if (const auto *keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                    /* close window if escape is pressed */
                    if (keyPressed->scancode == sf::Keyboard::Scancode::Escape) {
                        window.close();
                    }
                    /* clear canvas if q is pressed */
                    if (keyPressed->scancode == sf::Keyboard::Scancode::Q) {
                        for (auto &row: grid) {
                            std::fill(row.begin(), row.end(), 0);
                        }
                    }
                    /* if S is pressed, send grid to NN */
                    if (keyPressed->scancode == sf::Keyboard::Scancode::S) {
                        std::cout << "QUERY" << std::endl;
                        std::vector<float> input = downSampleMNIST(grid);
                        queryModel(nn, input);
                    }
                }
            }

            sf::Vector2i mousePos = sf::Mouse::getPosition(window);

            bool inBounds = (mousePos.x >= 0 && mousePos.x < CANVAS_WIDTH && 
                            mousePos.y >= 0 && mousePos.y < CANVAS_HEIGHT);

            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left) && inBounds) {
                int gridX = mousePos.x / CELL_SIZE;
                int gridY = mousePos.y / CELL_SIZE;
                grid[gridY][gridX] = 1; // Paint black
            }

            if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Right) && inBounds) {
                int gridX = mousePos.x / CELL_SIZE;
                int gridY = mousePos.y / CELL_SIZE;
                grid[gridY][gridX] = 0; // Erase to white
            }

            window.clear(sf::Color::White);

            for (int x = 0; x < GRID_WIDTH; x++) {
                for (int y = 0; y < GRID_HEIGHT; y++) {
                    if (grid[y][x] == 1) { // Only draw black cells
                        sf::RectangleShape cell(sf::Vector2f(CELL_SIZE, CELL_SIZE));
                        cell.setPosition(sf::Vector2f(x * CELL_SIZE, y * CELL_SIZE));
                        cell.setFillColor(sf::Color::Black);
                        window.draw(cell);
                    }
                }
            }

            window.display();
        }

    } catch (const std::exception &e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}