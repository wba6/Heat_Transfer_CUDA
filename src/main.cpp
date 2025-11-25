// main.cpp — SDL3 version (Windows/MSVC friendly)
#include <SDL3/SDL.h>
#include <cstdio>
#include <vector>
#include <algorithm>   // std::clamp
#include "heat.cuh"

// RAII cleanup
struct SdlCleanup {
    SDL_Window*   window{nullptr};
    SDL_Renderer* renderer{nullptr};
    SDL_Texture*  texture{nullptr};
    ~SdlCleanup() {
        if (texture)  SDL_DestroyTexture(texture);
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window)   SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

int main(int, char**) {
    const int   nx    = 512;
    const int   ny    = 512;
    const float alpha = 1.0f;
    const float dx    = 1.0f;
    const float dt    = 0.24f * dx * dx / (4.0f * alpha); // keep r<=0.25

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    SdlCleanup sdl;
    sdl.window = SDL_CreateWindow("CUDA Heat Simulation (SDL3)", nx, ny, SDL_WINDOW_RESIZABLE);
    if (!sdl.window) { std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError()); return 1; }

    sdl.renderer = SDL_CreateRenderer(sdl.window, nullptr);
    if (!sdl.renderer) { std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError()); return 1; }

    sdl.texture = SDL_CreateTexture(sdl.renderer, SDL_PIXELFORMAT_RGBA8888,
                                    SDL_TEXTUREACCESS_STREAMING, nx, ny);
    if (!sdl.texture) { std::fprintf(stderr, "SDL_CreateTexture failed: %s\n", SDL_GetError()); return 1; }

    // --- GPU sim state ---
    HeatSim sim;
    if (!heat_alloc(sim, nx, ny)) { std::fprintf(stderr, "Failed to allocate device buffers\n"); return 1; }

    // Hot disk init
    std::vector<float> h_init(nx * ny, 0.0f);
    const int cx = nx/2, cy = ny/2, r = nx/10;
    for (int j=-r; j<=r; ++j) for (int i=-r; i<=r; ++i) {
        int x = cx+i, y = cy+j;
        if (x>=0 && x<nx && y>=0 && y<ny && (i*i + j*j) <= r*r) h_init[y*nx + x] = 1.0f;
    }
    heat_upload(sim, h_init.data());

    std::vector<unsigned char> h_rgba(nx * ny * 4, 0);

    bool running = true;
    bool mouse_down = false;
    int  brush_radius = 10;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_EVENT_QUIT:
                running = false; break;
            case SDL_EVENT_KEY_DOWN:
                // SDL3: use ev.key.key (SDL_Keycode), not ev.key.keysym
                if (ev.key.key == SDLK_ESCAPE) running = false;
                break;
            case SDL_EVENT_MOUSE_BUTTON_DOWN:
                if (ev.button.button == SDL_BUTTON_LEFT) mouse_down = true;
                break;
            case SDL_EVENT_MOUSE_BUTTON_UP:
                if (ev.button.button == SDL_BUTTON_LEFT) mouse_down = false;
                break;
            case SDL_EVENT_MOUSE_WHEEL: {
                // SDL3 wheel.y is float; convert to an int step for clamp to be unambiguous
                int step = (ev.wheel.y > 0.0f) ? 1 : (ev.wheel.y < 0.0f ? -1 : 0);
                brush_radius = std::clamp(brush_radius + step, 1, 100);
                break;
            }
            default: break;
            }
        }

        if (mouse_down) {
            float mx = 0.0f, my = 0.0f;
            SDL_GetMouseState(&mx, &my); // SDL3 returns floats
            int ww = 0, wh = 0; SDL_GetWindowSize(sdl.window, &ww, &wh);
            int tx = std::clamp(static_cast<int>(mx * nx / std::max(1, ww)), 0, nx-1);
            int ty = std::clamp(static_cast<int>(my * ny / std::max(1, wh)), 0, ny-1);
            heat_paint(sim, tx, ty, brush_radius, 1.0f);
        }

        // A couple of steps per frame
        for (int k = 0; k < 2; ++k) heat_step(sim, alpha, dx, dt);

        // Map temps -> RGBA on device, copy back for SDL texture update
        heat_to_rgba(sim, h_rgba.data(), 0.0f, 1.0f);

        // Update and render
        const int pitch = nx * 4;
        SDL_UpdateTexture(sdl.texture, nullptr, h_rgba.data(), pitch);

        SDL_RenderClear(sdl.renderer);
        int ww = 0, wh = 0; SDL_GetWindowSize(sdl.window, &ww, &wh);
        SDL_FRect dst{0.f, 0.f, static_cast<float>(ww), static_cast<float>(wh)};
        SDL_RenderTexture(sdl.renderer, sdl.texture, nullptr, &dst);
        SDL_RenderPresent(sdl.renderer);
    }

    heat_free(sim);
    return 0;
}
