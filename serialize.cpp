#include "grayMatch.h"
#include "privateType.h"

class Buffer {
public:
    Buffer(int size_, unsigned char *data_)
        : m_size(size_)
        , m_data(data_) {}

    virtual void operator&(uchar &val)                   = 0;
    virtual void operator&(std::vector<cv::Mat> &val)    = 0;
    virtual void operator&(std::vector<cv::Scalar> &val) = 0;
    virtual void operator&(std::vector<double> &val)     = 0;
    virtual void operator&(std::vector<uchar> &val)      = 0;

    void operator&(Model &val) {
        this->operator&(val.pyramids);
        this->operator&(val.mean);
        this->operator&(val.normal);
        this->operator&(val.invArea);
        this->operator&(val.equal1);
        this->operator&(val.borderColor);
    }

    int count() const {
        return m_size;
    }

protected:
    int            m_size = 0;
    unsigned char *m_data = nullptr;
};

class WriteOperationBase {
public:
    virtual void write(void *dst, void *src, int size) = 0;
};


void binWrite(void *dst, void *src, int size) {
    memcpy(dst, src, size);
}


void fakeWrite(void *dst, void *src, int size)  {
    (void)(dst);
    (void)(src);
    (void)(size);
}

using Write = void(*)(void*,void*,int);

template <Write write> class OutBuffer : public Buffer {
public:
    OutBuffer(unsigned char *data_)
        : Buffer(0, data_) {}

    void operator&(uchar &val) final {
        write(m_data + m_size, &val, sizeof(val));
        m_size += static_cast<int>(sizeof(val));
    }
    void operator&(std::vector<cv::Mat> &val) final {
        int size = static_cast<int>(val.size());
        write(m_data + m_size, &size, sizeof(size));
        m_size += static_cast<int>(sizeof(size));

        for (auto &element : val) {
            writeElement(element);
        }
    }
    void writeElement(cv::Mat &val) {
        write(m_data + m_size, &val.cols, sizeof(int));
        m_size += static_cast<int>(sizeof(int));

        write(m_data + m_size, &val.rows, sizeof(int));
        m_size += static_cast<int>(sizeof(int));

        for (int i = 0; i < val.rows; i++) {
            write(m_data + m_size, val.ptr<unsigned char>(i), val.cols);
            m_size += val.cols;
        }
    }
    void operator&(std::vector<cv::Scalar> &val) final {
        int size = static_cast<int>(val.size());
        write(m_data + m_size, &size, sizeof(size));
        m_size += static_cast<int>(sizeof(size));

        for (auto &element : val) {
            writeElement(element);
        }
    }
    void writeElement(cv::Scalar &val) {
        write(m_data + m_size, val.val, sizeof(double) * 4);
        m_size += static_cast<int>(sizeof(double)) * 4;
    }
    void operator&(std::vector<double> &val) final {
        int size = static_cast<int>(val.size());
        write(m_data + m_size, &size, sizeof(size));
        m_size += static_cast<int>(sizeof(size));

        write(m_data + m_size, val.data(), static_cast<int>(sizeof(double)) * size);
        m_size += static_cast<int>(sizeof(double)) * size;
    }
    void operator&(std::vector<uchar> &val) final {
        int size = static_cast<int>(val.size());
        write(m_data + m_size, &size, sizeof(size));
        m_size += static_cast<int>(sizeof(size));

        write(m_data + m_size, val.data(), sizeof(uchar) * size);
        m_size += static_cast<int>(sizeof(uchar)) * size;
    }
};

using SizeCountBuffer = OutBuffer<fakeWrite>;
using WriteBuffer     = OutBuffer<binWrite>;

class ReadBuffer : public Buffer {
public:
    ReadBuffer(unsigned char *data_)
        : Buffer(0, data_) {}

    void operator&(uchar &val) final {
        memcpy(&val, m_data + m_size, sizeof(uchar));
        m_size += static_cast<int>(sizeof(uchar));
    }
    void operator&(std::vector<cv::Mat> &val) final {
        int count = 0;
        memcpy(&count, m_data + m_size, sizeof(int));
        val.resize(count);
        m_size += static_cast<int>(sizeof(count));

        for (auto &element : val) {
            read(element);
        }
    }
    void read(cv::Mat &val) {
        int width = 0;
        memcpy(&width, m_data + m_size, sizeof(int));
        m_size += static_cast<int>(sizeof(int));

        int height = 0;
        memcpy(&height, m_data + m_size, sizeof(int));
        m_size += static_cast<int>(sizeof(int));

        val     = cv::Mat(cv::Size(width, height), CV_8UC1, m_data + m_size);
        m_size += width * height;
    }
    void operator&(std::vector<cv::Scalar> &val) final {
        int count = 0;
        memcpy(&count, m_data + m_size, sizeof(int));
        val.resize(count);
        m_size += static_cast<int>(sizeof(count));

        for (auto &element : val) {
            read(element);
        }
    }
    void read(cv::Scalar &val) {
        memcpy(val.val, m_data + m_size, sizeof(double) * 4);
        m_size += static_cast<int>(sizeof(double)) * 4;
    }
    void operator&(std::vector<double> &val) final {
        int count = 0;
        memcpy(&count, m_data + m_size, sizeof(int));
        val.resize(count);
        m_size += static_cast<int>(sizeof(count));

        memcpy(val.data(), m_data + m_size, sizeof(double) * count);
        m_size += static_cast<int>(sizeof(double) )* count;
    }
    void operator&(std::vector<uchar> &val) final {
        int count = 0;
        memcpy(&count, m_data + m_size, sizeof(int));
        val.resize(count);
        m_size += static_cast<int>(sizeof(count));

        memcpy(val.data(), m_data + m_size, sizeof(bool) * count);
        m_size += static_cast<int>(sizeof(uchar)) * count;
    }
};

void operation(Buffer *buf, Model &model) {
    (*buf) & (model);
}

bool serialize(const Model_t model, unsigned char *buffer, int *size) {
    if (nullptr == size) {
        return false;
    }

    if (nullptr == model) {
        *size = 0;
        return false;
    }

    SizeCountBuffer countor(buffer);
    operation(&countor, *model);
    *size = countor.count();

    if (nullptr == buffer) {
        return true;
    }

    if (countor.count() > *size) {
        *size = 0;
        return false;
    }

    WriteBuffer writer(buffer);
    operation(&writer, *model);
    return true;
}

Model_t deserialize(unsigned char *buffer, int size) {
    if (size < 1 || nullptr == buffer) {
        return nullptr;
    }

    ReadBuffer reader(buffer);
    auto      *model = new Model;
    operation(&reader, *model);

    return model;
}
