#include <duda/random/normal.hpp>

#include <iostream>

int main()
{
    std::cout << "original fmt: " << 0.123456789 << "\n";

    {
        std::cout << "\nvector with default print precision:\n";

        auto x = duda::random_normal<float>(8);

        for (int i = 0; i < x.size(); ++i)
        {
            x.set(i, i);
        }
        std::cout << x;
    }

    {
        std::cout << "\nmatrix with print precision 2:\n";

        auto x = duda::random_normal<double>(4, 5);

        for (int i = 0; i < x.rows(); ++i)
        {
            for (int j = 0; j < x.cols(); ++j)
            {
                x.set(i, j, j + i * 1000.0);
            }
        }
        duda::print_precision().value() = 2;
        std::cout << x;
    }

    std::cout << "\noriginal fmt is OK: " << 0.123456789 << "\n";

    return 0;
}
