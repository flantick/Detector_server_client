#include <iostream>
#include <boost/asio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/types_c.h>

using boost::asio::ip::tcp;

int main(int argc, char* argv[]) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: ./client <server_ip> <server_port> <video_source> [output]" << std::endl;
        return 1;
    }

    std::string serverIP = argv[1];
    std::string serverPort = argv[2];
    cv::String source = argv[3];
    cv::String output = " ";
    bool record = false;

    if (argc == 5) {
        output = argv[4];
        record = true;
        
    }

    try {
        boost::asio::io_context io_context;
        // Create a resolver object to resolve the server address
        tcp::resolver resolver(io_context);
        tcp::resolver::results_type endpoints = resolver.resolve(serverIP, serverPort);

        // Create a socket object to connect to the server
        tcp::socket socket(io_context);
        boost::asio::connect(socket, endpoints);

        // prepare opencv parameters
        cv::VideoCapture cap = (source != "0") ? cv::VideoCapture(source) : cv::VideoCapture(0);
        double fps;
        cv::Size frameSize;
        int rec_fourcc;
        cv::VideoWriter rec;
        if (record) {
           fps = cap.get(cv::CAP_PROP_FPS);
            frameSize = {
            640,640
            };
     
            if (output.length() > 0) {
                rec_fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                rec.open(output, rec_fourcc, fps, frameSize);
            }
         }

        // Send frames to the server
        cv::Mat frame;
        while (cap.isOpened()) {
            cap >> frame;


            // Check for an empty frame
            if (frame.empty()) {
                std::cerr << "Empty frame" << std::endl;
                continue;
            }

            // Encode the frame into a serialized buffer data
            std::vector<uchar> buffer;
            cv::imencode(".jpg", frame, buffer);

            // Send the buffer size to the server
            size_t buffer_size = buffer.size();
            boost::asio::write(socket, boost::asio::buffer(&buffer_size, sizeof(buffer_size)));

            // Send the buffer data to the server
            boost::asio::write(socket, boost::asio::buffer(buffer.data(), buffer_size));

            // Receive the buffer size from the server
            size_t received_buffer_size;
            boost::asio::read(socket, boost::asio::buffer(&received_buffer_size, sizeof(received_buffer_size)));

            // Receive the buffer data from the server
            std::vector<uchar> received_buffer(received_buffer_size);
            boost::asio::read(socket, boost::asio::buffer(received_buffer));

            // Decode the buffer data into a frame
            cv::Mat received_frame = cv::imdecode(received_buffer, cv::IMREAD_COLOR);

            // Check for successful frame decoding / exit the program
            if (received_frame.empty() || cv::waitKey(1) == 'c')
                break;

            // Record the received frame
            if (rec.isOpened())
                rec.write(received_frame);

            // Displaying the received frame
            cv::imshow("PRESS C TO EXIT", received_frame);
            cv::waitKey(1);


        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}
