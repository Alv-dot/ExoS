#include <mlpack/core.hpp>
#include <mlpack/methods/svm/svm.hpp>

using namespace mlpack;
using namespace mlpack::svm;
using namespace arma;

int main() {
    mat trainingData;
    Row<size_t> labels;

    data::Load("training_data.csv", trainingData, true);
    data::Load("training_labels.csv", labels, true);

    SVM<> svm(trainingData, labels, 6, 0.01);

    data::Save("svm_model.xml", "svm", svm);

    std::cout << "SVM Model Retrained and Saved!" << std::endl;
    return 0;
}
