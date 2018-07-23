#include <gtest/gtest.h>

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    //::testing::AddGlobalTestEnvironment(new CLEnvironment);

    int result = RUN_ALL_TESTS();
    std::cin.get();
    return result;
}
