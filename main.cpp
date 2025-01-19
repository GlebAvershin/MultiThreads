#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <stdexcept>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <cstring>
#include <chrono>
#include <cassert>  // Для assert

struct Task {
    int startRow;
    int endRow;
    bool isEnd; 
    
    Task(int s, int e): startRow(s), endRow(e), isEnd(false) {}
    Task(): startRow(0), endRow(0), isEnd(true) {} 
};

template<typename T>
class BlockingQueue {
public:
    BlockingQueue(size_t capacity) : capacity_(capacity) {}

    void push(const T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_producer_.wait(lock, [this] { return queue_.size() < capacity_; });
        queue_.push(item);
        cond_consumer_.notify_one();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_consumer_.wait(lock, [this] { return !queue_.empty(); });
        T item = queue_.front();
        queue_.pop();
        cond_producer_.notify_one();
        return item;
    }

private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_producer_;
    std::condition_variable cond_consumer_;
    size_t capacity_;
};

void invertBlock(cv::Mat& image, int startRow, int endRow) {
    for (int r = startRow; r <= endRow; ++r) {
        uchar* rowPtr = image.ptr<uchar>(r);
        for (int c = 0; c < image.cols * image.channels(); ++c) {
            rowPtr[c] = 255 - rowPtr[c];
        }
    }
}

// Проверка по расширению файла, является ли он изображением
bool isImageFile(const std::filesystem::path& p) {
    if (!p.has_extension()) return false;
    // Список поддерживаемых расширений можно дополнять
    std::string ext = p.extension().string();
    for (auto & c: ext) c = std::tolower(static_cast<unsigned char>(c));
    return (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff");
}

// тест для функции инверсии блоков
void testInvertBlock() {
    cv::Mat image(4, 4, CV_8UC3, cv::Scalar(0, 0, 0));  // Черное изображение 4x4
    invertBlock(image, 0, 3);
    
    // Проверяем, что все пиксели стали белыми (255, 255, 255)
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            uchar* pixel = image.ptr<uchar>(r, c);
            assert(pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255);
        }
    }
    std::cout << "testInvertBlock passed\n";
}

// тест для обработки изображения
void testIsImageFile() {
    assert(isImageFile("test.jpg") == true);
    assert(isImageFile("test.jpeg") == true);
    assert(isImageFile("test.png") == true);
    assert(isImageFile("test.bmp") == true);
    assert(isImageFile("test.tiff") == true);
    assert(isImageFile("test.txt") == false);  // Это не изображение
    std::cout << "testIsImageFile passed\n";
}

bool processImage(const std::string& inputImagePath, const std::string& outputDir, int numThreads, int capacity, int blocks) {
    cv::Mat inputImage = cv::imread(inputImagePath);
    if (inputImage.empty()) {
        std::cerr << "Не удалось загрузить изображение: " << inputImagePath << "\n";
        return false;
    }

    int rows = inputImage.rows;
    if (blocks > rows) {
        std::cerr << "Количество блоков больше количества строк изображения. Уменьшите число блоков.\n";
        return false;
    }

    int blockHeight = rows / blocks;
    int remainder = rows % blocks;

    cv::Mat resultImage = inputImage.clone();

    BlockingQueue<Task> taskQueue(capacity);
    std::atomic<bool> producerDone(false);
    std::atomic<int> tasksRemaining(blocks);

    // Поток-производитель
    std::thread producerThread([&](){
        int start = 0;
        for (int i = 0; i < blocks; ++i) {
            int currentBlockHeight = blockHeight + (i < remainder ? 1 : 0);
            int end = start + currentBlockHeight - 1;
            Task task(start, end);
            taskQueue.push(task);
            start = end + 1;
        }

        for (int i = 0; i < numThreads; ++i) {
            taskQueue.push(Task()); // сигнал о завершении
        }

        producerDone = true;
    });

    // Потоки-потребители
    std::vector<std::thread> consumerThreads;
    consumerThreads.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        consumerThreads.emplace_back([&](){
            while (true) {
                Task task = taskQueue.pop();
                if (task.isEnd) {
                    break;
                }
                invertBlock(resultImage, task.startRow, task.endRow);
                tasksRemaining.fetch_sub(1);
            }
        });
    }

    producerThread.join();
    for (auto& th : consumerThreads) {
        th.join();
    }

    if (tasksRemaining.load() != 0) {
        std::cerr << "Не все задачи были обработаны!\n";
        return false;
    }

    std::filesystem::path inputPath(inputImagePath);
    std::string outputFileName = inputPath.stem().string() + "_processed" + inputPath.extension().string();
    std::filesystem::path outputPath = std::filesystem::path(outputDir) / outputFileName;

    if (!cv::imwrite(outputPath.string(), resultImage)) {
        std::cerr << "Не удалось сохранить результат в: " << outputPath.string() << "\n";
        return false;
    }

    std::cout << "Обработка завершена. Результат: " << outputPath.string() << std::endl;
    return true;
}

int main(int argc, char** argv) {
    // Запуск тестов
    testInvertBlock();
    testIsImageFile();

    auto startTime = std::chrono::high_resolution_clock::now();

    std::string inputDir;
    std::string outputDir;
    int numThreads = 4;
    int capacity = 4;
    int blocks = 4;

    bool outputSet = false;
    bool inputSet = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            inputDir = argv[++i];
            inputSet = true;
        } else if (std::strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            outputDir = argv[++i];
            outputSet = true;
        } else if (std::strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            int t = std::atoi(argv[++i]);
            if (t == 1 || t == 2 || t == 4 || t == 8) {
                numThreads = t;
            } else {
                std::cerr << "Неправильное значение для -t. Используйте 1, 2, 4 или 8.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            capacity = std::atoi(argv[++i]);
            if (capacity <= 0) {
                std::cerr << "Ёмкость очереди должна быть положительным числом.\n";
                return 1;
            }
        } else if (std::strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            blocks = std::atoi(argv[++i]);
            if (blocks <= 0) {
                std::cerr << "Количество блоков должно быть положительным числом.\n";
                return 1;
            }
        } else {
            std::cerr << "Неизвестный аргумент: " << argv[i] << "\n";
            return 1;
        }
    }

    if (!inputSet) {
        std::cerr << "Не указана входная папка с изображениями. Используйте -i /path/to/input_folder\n";
        return 1;
    }

    if (!outputSet) {
        std::cerr << "Не указана выходная папка. Используйте -o /path/to/output_folder\n";
        return 1;
    }

    if (!std::filesystem::exists(inputDir)) {
        std::cerr << "Входная папка не существует: " << inputDir << "\n";
        return 1;
    }

    // Создаём выходную папку
    std::filesystem::create_directories(outputDir);

    // Собираем все изображения из входной папки
    std::vector<std::string> imagePaths;
    for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && isImageFile(entry.path())) {
            imagePaths.push_back(entry.path().string());
        }
    }

    if (imagePaths.empty()) {
        std::cerr << "Не найдено изображений в папке: " << inputDir << "\n";
        return 1;
    }

    // Обрабатываем все изображения по очереди
    for (const auto& imgPath : imagePaths) {
        std::cout << "Обработка изображения: " << imgPath << std::endl;
        if (!processImage(imgPath, outputDir, numThreads, capacity, blocks)) {
            std::cerr << "Ошибка при обработке: " << imgPath << "\n";
        }
    }

    // Замер конца времени
    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = endTime - startTime;
    double seconds = elapsed.count();

    std::cout << "\nВсе изображения обработаны.\n";
    std::cout << "Общее время обработки: " << seconds << " сек.\n";

    return 0;
}

// ./main -i /home/user1/Images/ -o /home/user1/output -t 2 -c 2 -b 2
