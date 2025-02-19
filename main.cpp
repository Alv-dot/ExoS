#include <Eigen/Dense>
#include <boost/asio.hpp>
#include <mlpack/methods/svm/svm.hpp>
#include <mlpack/core/data/load.hpp>
#include <vector>
#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>

using namespace mlpack;
using namespace mlpack::svm;
using namespace arma;

// ---------------- Feature Extraction Functions ----------------

// Compute Mean Absolute Value (MAV)
double computeMAV(const std::vector<double>& signal) {
    Eigen::VectorXd vec = Eigen::Map<const Eigen::VectorXd>(signal.data(), signal.size());
    return vec.array().abs().mean();
}

// Compute Zero Crossings (ZC)
int computeZC(const std::vector<double>& signal) {
    int count = 0;
    for (size_t i = 1; i < signal.size(); ++i) {
        if ((signal[i - 1] > 0 && signal[i] < 0) || (signal[i - 1] < 0 && signal[i] > 0)) {
            count++;
        }
    }
    return count;
}

// Compute Slope Sign Changes (SSC)
int computeSSC(const std::vector<double>& signal) {
    int count = 0;
    for (size_t i = 2; i < signal.size(); ++i) {
        if ((signal[i] - signal[i - 1]) * (signal[i - 1] - signal[i - 2]) < 0) {
            count++;
        }
    }
    return count;
}

// Compute Waveform Length (WL)
double computeWL(const std::vector<double>& signal) {
    double length = 0.0;
    for (size_t i = 1; i < signal.size(); ++i) {
        length += std::abs(signal[i] - signal[i - 1]);
    }
    return length;
}

// Compute RMS Value
double computeRMS(const std::vector<double>& signal) {
    if (signal.empty()) return 0.0;
    Eigen::VectorXd vec = Eigen::Map<const Eigen::VectorXd>(signal.data(), signal.size());
    return std::sqrt(vec.array().square().mean());
}

// ---------------- Exoskeleton Control Function ----------------

void controlExoskeleton(size_t predictedMovement) {
    switch (predictedMovement) {
        case 0: std::cout << "Action: Relax" << std::endl; break;
        case 1: std::cout << "Action: Grip Object" << std::endl; break;
        case 2: std::cout << "Action: Lift Arm" << std::endl; break;
        case 3: std::cout << "Action: Extend Arm" << std::endl; break;
        case 4: std::cout << "Action: Rotate Wrist" << std::endl; break;
        case 5: std::cout << "Action: Bend Elbow" << std::endl; break;
        default: std::cout << "Unknown Action" << std::endl;
    }
}

// ---------------- Main Program ----------------

int main() {
    boost::asio::io_service io;
    boost::asio::serial_port serial(io, "COM3"); // Replace with correct port
    serial.set_option(boost::asio::serial_port_base::baud_rate(115200));

    SVM<> svm;
    data::Load("svm_model.xml", "svm", svm);

    std::vector<char> buffer(1024);
    std::ofstream logFile("performance_log.csv", std::ios::app);
    logFile << "Time,Prediction,ProcessingTime(ms)" << std::endl;

    std::ofstream trainFile("training_data.csv", std::ios::app);
    trainFile << "MAV,ZC,SSC,WL,RMS,Label" << std::endl;

    try {
        while (true) {
            auto startTime = std::chrono::high_resolution_clock::now();

            boost::asio::read(serial, boost::asio::buffer(buffer));

            std::vector<double> emgSignal;
            for (char c : buffer) {
                emgSignal.push_back(static_cast<double>(static_cast<unsigned char>(c)));
            }

            // Extract Features
            double mav = computeMAV(emgSignal);
            int zc = computeZC(emgSignal);
            int ssc = computeSSC(emgSignal);
            double wl = computeWL(emgSignal);
            double rms = computeRMS(emgSignal);

            vec features = {mav, static_cast<double>(zc), static_cast<double>(ssc), wl, rms};

            // Predict Movement
            size_t prediction = svm.Classify(features);

            // Log Performance
            auto endTime = std::chrono::high_resolution_clock::now();
            double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
            logFile << processingTime << "," << prediction << "," << processingTime << std::endl;

            // Store Training Data
            size_t groundTruth = 0;  
            trainFile << mav << "," << zc << "," << ssc << "," << wl << "," << rms << "," << groundTruth << std::endl;

            controlExoskeleton(prediction);
        }
    } catch (const boost::system::system_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    logFile.close();
    trainFile.close();
    return 0;
}
