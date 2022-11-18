#include "Vulkan.h"
#include <stdexcept>
#include <iostream>

int main()
{
	Application VulkanApp;

	try
	{
		VulkanApp.Run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
