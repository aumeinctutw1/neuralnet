# neuralnet
Neuralnetwork from the book "Make your own Neuralnetwork" implemented in c++
SFML is used to display a 400x400 drawing canvas, the input from the canvas is downsampled to
a 28x28 Grid, which then is used as input to the neuralnet

Press Q -> clear canvas
Press S -> query neuralnetwork 
Press ESC -> Close the window
Left Mouse Button -> Draw black pixels on the canvas
Right Mousr Button -> Erase black pixels on the canvas

# build
make 
make clean && make

# run
./nn 

# Output
![Alt text](/screenshot.png?raw=true "Optional Title")