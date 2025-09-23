#ifndef PYROCLAST_TURBO_UTILS_HPP
#define PYROCLAST_TURBO_UTILS_HPP

namespace pyroclast_turbo {

// Clamp integer to [lo, hi]
static inline int clip_int(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

} // namespace pyroclast_turbo

#endif // PYROCLAST_TURBO_UTILS_HPP
