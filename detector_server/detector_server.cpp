
#include <iostream>
#include <boost/asio.hpp>
#include <thread>
#include "utilities.h"


using boost::asio::ip::tcp;
using namespace std;


void handleClient(tcp::socket socket,
    const std::string& source
) {
    torch::jit::script::Module model = torch::jit::load(source);
    try {
        boost::asio::streambuf buf;

        while (true) {
            // Receive the buffer size from the client
            size_t buffer_size;
            boost::asio::read(socket, boost::asio::buffer(&buffer_size, sizeof(buffer_size)));

            // Receive the buffer data from the client
            std::vector<uchar> buffer(buffer_size);
            boost::asio::read(socket, boost::asio::buffer(buffer));

            // Decode the buffer data into a frame
            cv::Mat received_frame = cv::imdecode(buffer, cv::IMREAD_COLOR);

            // Check if the frame decoding was successful
            if (received_frame.empty()) {
                std::cerr << "Failed to decode received frame" << std::endl;
                continue;
            }

            // Process the frame using the model
            cv::Mat frame = detect(model, received_frame, 640);

            // Encode the frame into a serialized buffer data
            std::vector<uchar> out_buffer;
            cv::imencode(".jpg", frame, out_buffer);

            // Send the buffer size to the client
            size_t out_buffer_size = out_buffer.size();
            boost::asio::write(socket, boost::asio::buffer(&out_buffer_size, sizeof(out_buffer_size)));

            // Send the buffer data to the client
            boost::asio::write(socket, boost::asio::buffer(out_buffer));
        }

        // Close the connection with the client
        socket.close();
    }
    catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <ip> <port>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string ip = argv[2];
    int port = std::stoi(argv[3]);

    try {
        boost::asio::io_context io_context;

        // Create an acceptor object to listen for incoming connections
        tcp::acceptor acceptor(io_context, tcp::endpoint(boost::asio::ip::make_address(ip), port));
        std::cout << "SERVER STARTED" << std::endl;

        std::vector<std::thread> threads;

        while (true) {
            // Wait for and accept an incoming connection
            tcp::socket socket(io_context);
            acceptor.accept(socket);

            // Create a new thread to handle the client
            threads.emplace_back(handleClient, std::move(socket), modelPath);
        }

        // Wait for all threads to finish
        for (auto& thread : threads) {
            thread.join();
        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    std::cout << "SERVER OFF" << std::endl;
    return 0;
}
