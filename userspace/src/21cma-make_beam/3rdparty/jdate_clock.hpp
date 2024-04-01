// jdate_clock by Howard Hinnant
// https://stackoverflow.com/questions/33964461/handling-julian-days-in-c11-14
#pragma once
#ifndef JDATE_CLOCK_HPP
#define JDATE_CLOCK_HPP

#include <chrono>

struct jdate_clock;

template <class Duration>
    using jdate_time = std::chrono::time_point<jdate_clock, Duration>;

struct jdate_clock
{
    using rep        = double;
    using period     = std::chrono::days::period;
    using duration   = std::chrono::duration<rep, period>;
    using time_point = std::chrono::time_point<jdate_clock>;

    static constexpr bool is_steady = false;

    // static time_point now() noexcept;

    template <class Duration>
    static
    auto
    from_sys(std::chrono::sys_time<Duration> const& tp) noexcept;

    // template <class Duration>
    // static
    // auto
    // to_sys(jdate_time<Duration> const& tp) noexcept;
};

template <class Duration>
auto
jdate_clock::from_sys(std::chrono::sys_time<Duration> const& tp) noexcept
{
    using namespace std;
    using namespace chrono;
    auto constexpr epoch = sys_days{November/24/-4713} + 12h;
    using ddays = std::chrono::duration<long double, days::period>;
    if constexpr (sys_time<ddays>{sys_time<Duration>::min()} < sys_time<ddays>{epoch})
    {
        return jdate_time<Duration>{tp - epoch};
    }
    else
    {
        // Duration overflows at the epoch.  Sub in new Duration that won't overflow.
        using D = std::chrono::duration<int64_t, ratio<1, 10'000'000>>;
        return jdate_time<Duration>{round<D>(tp) - epoch};
    }
}

// template <class Duration>
// auto
// jdate_clock::to_sys(jdate_time<Duration> const& tp) noexcept
// {
//     using namespace std::chrono;
//     return sys_time{tp - clock_cast<jdate_clock>(sys_days{})};
// }

// jdate_clock::time_point
// jdate_clock::now() noexcept
// {
//     using namespace std::chrono;
//     return clock_cast<jdate_clock>(system_clock::now());
// }

#endif  // JDATE_CLOCK_HPP
