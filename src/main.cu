#include <SDL3/SDL.h>
#include <iostream>

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    // Create a window
    SDL_Window* window = SDL_CreateWindow(
        "SDL3 Heat Transfer Window", // Window title
        800,                  // Window width
        600,                  // Window height
        0                     // No specific flags for this basic example
    );

    if (window == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    // create render
    SDL_Renderer* renderer = SDL_CreateRenderer(window, nullptr);
    if (renderer) {
        SDL_SetRenderDrawColor(renderer, 200, 0, 0, 255);
        std::cout << "Render created" << std::endl;
    }

    // Event loop
    bool quit = false;
    SDL_Event e;
    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_EVENT_QUIT) {
                quit = true;
            }
        }
        // clear render buffer
        SDL_RenderClear(renderer);
        // render new stuff
        SDL_RenderPresent(renderer);
    }

    // Destroy window and quit SDL
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}