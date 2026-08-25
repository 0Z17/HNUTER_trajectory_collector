#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/vector.h>

#include "ompl/base/StateSampler.h"
#include "ompl/base/StateSpace.h"
#include "ompl/base/spaces/SE3StateSpace.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
namespace ob = ompl::base;

namespace
{
struct Quaternion
{
    double w;
    double x;
    double y;
    double z;
};

Quaternion normalized(Quaternion value)
{
    const double norm = std::sqrt(
        value.w * value.w + value.x * value.x +
        value.y * value.y + value.z * value.z
    );
    if (!(norm > 1.0e-12) || !std::isfinite(norm))
        throw std::invalid_argument("region quaternion must be finite and nonzero");
    value.w /= norm;
    value.x /= norm;
    value.y /= norm;
    value.z /= norm;
    return value;
}

Quaternion multiply(const Quaternion &left, const Quaternion &right)
{
    return normalized({
        left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z,
        left.w * right.x + left.x * right.w + left.y * right.z - left.z * right.y,
        left.w * right.y - left.x * right.z + left.y * right.w + left.z * right.x,
        left.w * right.z + left.x * right.y - left.y * right.x + left.z * right.w,
    });
}

Quaternion fromEuler(double roll, double pitch, double yaw)
{
    const double cr = std::cos(0.5 * roll);
    const double sr = std::sin(0.5 * roll);
    const double cp = std::cos(0.5 * pitch);
    const double sp = std::sin(0.5 * pitch);
    const double cy = std::cos(0.5 * yaw);
    const double sy = std::sin(0.5 * yaw);
    return normalized({
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    });
}

std::array<double, 3> rotate(
    const Quaternion &quaternion,
    const std::array<double, 3> &point
)
{
    const double xx = quaternion.x * quaternion.x;
    const double yy = quaternion.y * quaternion.y;
    const double zz = quaternion.z * quaternion.z;
    const double xy = quaternion.x * quaternion.y;
    const double xz = quaternion.x * quaternion.z;
    const double yz = quaternion.y * quaternion.z;
    const double wx = quaternion.w * quaternion.x;
    const double wy = quaternion.w * quaternion.y;
    const double wz = quaternion.w * quaternion.z;
    return {
        (1.0 - 2.0 * (yy + zz)) * point[0] +
            2.0 * (xy - wz) * point[1] + 2.0 * (xz + wy) * point[2],
        2.0 * (xy + wz) * point[0] +
            (1.0 - 2.0 * (xx + zz)) * point[1] + 2.0 * (yz - wx) * point[2],
        2.0 * (xz - wy) * point[0] + 2.0 * (yz + wx) * point[1] +
            (1.0 - 2.0 * (xx + yy)) * point[2],
    };
}

struct Region
{
    std::array<double, 3> center;
    std::array<double, 3> size;
    Quaternion positionQuaternion;
    Quaternion attitudeQuaternion;
    std::array<double, 3> attitudeJitter;
};

struct SamplerStatistics
{
    std::atomic<std::uint64_t> regional{0};
    std::atomic<std::uint64_t> uniform{0};
    std::atomic<std::uint64_t> rejectedRegional{0};
    std::atomic<std::uint64_t> allocations{0};
};

struct SamplerConfiguration
{
    std::vector<Region> regions;
    double regionalProbability;
    std::uint64_t seed;
    std::shared_ptr<SamplerStatistics> statistics;
};

class RegionBiasedSE3StateSampler final : public ob::StateSampler
{
public:
    RegionBiasedSE3StateSampler(
        const ob::StateSpace *space,
        std::shared_ptr<const SamplerConfiguration> configuration,
        std::uint64_t allocationIndex
    )
      : ob::StateSampler(space),
        configuration_(std::move(configuration)),
        fallback_(space->allocDefaultStateSampler()),
        generator_(configuration_->seed + 0x9e3779b97f4a7c15ULL * allocationIndex),
        unit_(0.0, 1.0)
    {
    }

    void sampleUniform(ob::State *state) override
    {
        if (unit_(generator_) >= configuration_->regionalProbability)
        {
            configuration_->statistics->uniform.fetch_add(1, std::memory_order_relaxed);
            fallback_->sampleUniform(state);
            return;
        }

        std::uniform_int_distribution<std::size_t> choose(
            0, configuration_->regions.size() - 1
        );
        for (unsigned int attempt = 0; attempt < 48; ++attempt)
        {
            const Region &region = configuration_->regions[choose(generator_)];
            std::array<double, 3> local{};
            std::array<double, 3> jitter{};
            for (std::size_t axis = 0; axis < 3; ++axis)
            {
                local[axis] = (unit_(generator_) - 0.5) * region.size[axis];
                jitter[axis] = (2.0 * unit_(generator_) - 1.0) *
                    region.attitudeJitter[axis];
            }
            const auto offset = rotate(region.positionQuaternion, local);
            auto *se3 = state->as<ob::SE3StateSpace::StateType>();
            se3->setXYZ(
                region.center[0] + offset[0],
                region.center[1] + offset[1],
                region.center[2] + offset[2]
            );
            const Quaternion attitude = multiply(
                region.attitudeQuaternion,
                fromEuler(jitter[0], jitter[1], jitter[2])
            );
            se3->rotation().w = attitude.w;
            se3->rotation().x = attitude.x;
            se3->rotation().y = attitude.y;
            se3->rotation().z = attitude.z;
            if (space_->satisfiesBounds(state))
            {
                configuration_->statistics->regional.fetch_add(
                    1, std::memory_order_relaxed
                );
                return;
            }
            configuration_->statistics->rejectedRegional.fetch_add(
                1, std::memory_order_relaxed
            );
        }
        configuration_->statistics->uniform.fetch_add(1, std::memory_order_relaxed);
        fallback_->sampleUniform(state);
    }

    void sampleUniformNear(
        ob::State *state,
        const ob::State *near,
        double distance
    ) override
    {
        fallback_->sampleUniformNear(state, near, distance);
    }

    void sampleGaussian(
        ob::State *state,
        const ob::State *mean,
        double standardDeviation
    ) override
    {
        fallback_->sampleGaussian(state, mean, standardDeviation);
    }

private:
    std::shared_ptr<const SamplerConfiguration> configuration_;
    ob::StateSamplerPtr fallback_;
    std::mt19937_64 generator_;
    std::uniform_real_distribution<double> unit_;
};

Region parseRegion(const std::vector<double> &values)
{
    if (values.size() != 17)
        throw std::invalid_argument("each flattened SE(3) region must contain 17 values");
    Region region{
        {values[0], values[1], values[2]},
        {values[3], values[4], values[5]},
        normalized({values[6], values[7], values[8], values[9]}),
        normalized({values[10], values[11], values[12], values[13]}),
        {values[14], values[15], values[16]},
    };
    for (double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument("SE(3) region values must be finite");
    for (std::size_t axis = 0; axis < 3; ++axis)
    {
        if (!(region.size[axis] > 0.0))
            throw std::invalid_argument("SE(3) region sizes must be positive");
        if (region.attitudeJitter[axis] < 0.0)
            throw std::invalid_argument("attitude jitter must be non-negative");
    }
    return region;
}
}  // namespace

NB_MODULE(_ompl_region_sampler, module)
{
    nb::class_<SamplerStatistics>(module, "SamplerStatistics")
        .def_prop_ro("regional_sample_count", [](const SamplerStatistics &stats)
        {
            return stats.regional.load(std::memory_order_relaxed);
        })
        .def_prop_ro("uniform_sample_count", [](const SamplerStatistics &stats)
        {
            return stats.uniform.load(std::memory_order_relaxed);
        })
        .def_prop_ro("rejected_regional_sample_count", [](const SamplerStatistics &stats)
        {
            return stats.rejectedRegional.load(std::memory_order_relaxed);
        })
        .def_prop_ro("sampler_allocation_count", [](const SamplerStatistics &stats)
        {
            return stats.allocations.load(std::memory_order_relaxed);
        });

    module.def(
        "install_region_state_sampler",
        [](ob::StateSpace &space,
           const std::vector<std::vector<double>> &flattenedRegions,
           double regionalProbability,
           std::uint64_t seed)
        {
            if (dynamic_cast<ob::SE3StateSpace *>(&space) == nullptr)
                throw std::invalid_argument("regional sampler requires SE3StateSpace");
            if (flattenedRegions.empty())
                throw std::invalid_argument("at least one sampling region is required");
            if (!(regionalProbability > 0.0 && regionalProbability <= 1.0))
                throw std::invalid_argument("regional probability must lie in (0, 1]");

            auto configuration = std::make_shared<SamplerConfiguration>();
            configuration->regionalProbability = regionalProbability;
            configuration->seed = seed;
            configuration->statistics = std::make_shared<SamplerStatistics>();
            configuration->regions.reserve(flattenedRegions.size());
            for (const auto &values : flattenedRegions)
                configuration->regions.push_back(parseRegion(values));

            space.setStateSamplerAllocator(
                [configuration](const ob::StateSpace *source)
                {
                    const std::uint64_t allocation =
                        configuration->statistics->allocations.fetch_add(
                            1, std::memory_order_relaxed
                        ) + 1;
                    return std::make_shared<RegionBiasedSE3StateSampler>(
                        source, configuration, allocation
                    );
                }
            );
            return configuration->statistics;
        },
        nb::arg("space"),
        nb::arg("flattened_regions"),
        nb::arg("regional_probability"),
        nb::arg("seed")
    );
}
