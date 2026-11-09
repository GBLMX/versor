#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <vsr/runtime.hpp>

namespace py = pybind11;
using namespace vsr::runtime;

namespace {

py::array_t<double> to_numpy_3(const std::array<double, 3>& values) {
    py::array_t<double> out(3);
    auto view = out.mutable_unchecked<1>();
    view(0) = values[0];
    view(1) = values[1];
    view(2) = values[2];
    return out;
}

py::array_t<double> to_numpy_4(const std::array<double, 4>& values) {
    py::array_t<double> out(4);
    auto view = out.mutable_unchecked<1>();
    view(0) = values[0];
    view(1) = values[1];
    view(2) = values[2];
    view(3) = values[3];
    return out;
}

py::array_t<double> coefficients_numpy(const Multivector& value) {
    const auto& coefficients = value.coefficients();
    py::array_t<double> out(static_cast<py::ssize_t>(coefficients.size()));
    auto view = out.mutable_unchecked<1>();
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
        view(static_cast<py::ssize_t>(i)) = coefficients[i];
    }
    return out;
}

std::string repr_multivector(const Multivector& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

py::array_t<double> rotate_many(
    py::array_t<double, py::array::c_style | py::array::forcecast> vectors,
    const Multivector& rotor) {
    const auto info = vectors.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::invalid_argument("vectors must have shape (N, 3)");
    }
    const py::ssize_t count = info.shape[0];
    py::array_t<double> out({count, static_cast<py::ssize_t>(3)});
    auto input = vectors.unchecked<2>();
    auto output = out.mutable_unchecked<2>();
    std::exception_ptr failure;
    {
        // The batch touches only C++ and NumPy-owned buffers. Large arrays no
        // longer hold Blender's GIL while rotating.
        py::gil_scoped_release release;
        try {
            for (py::ssize_t i = 0; i < count; ++i) {
                const Multivector image = E3::rotate(
                    E3::vec(input(i, 0), input(i, 1), input(i, 2)), rotor);
                output(i, 0) = image.get(1);
                output(i, 1) = image.get(2);
                output(i, 2) = image.get(4);
            }
        } catch (...) {
            failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
    return out;
}

} // namespace

PYBIND11_MODULE(vsr_runtime, m) {
    m.doc() = "PyBind11 bindings for the versor C++ runtime Clifford algebra core";
    m.attr("__version__") = "0.1.0";

    py::class_<Signature>(m, "Signature")
        .def(py::init<>())
        .def(py::init<std::vector<int>, int>(), py::arg("metric"), py::arg("name_offset") = 0)
        .def_static("of", [](const std::vector<int>& metric) { return Signature::of(metric); })
        .def_static("cl", &Signature::cl)
        .def_static("euclidean", &Signature::euclidean)
        .def_static("e3", &Signature::e3)
        .def_static("pga3", &Signature::pga3)
        .def_static("sta", &Signature::sta)
        .def_static("cga3", &Signature::cga3)
        .def_property_readonly("n", &Signature::n)
        .def_property_readonly("size", &Signature::size)
        .def("metric", &Signature::metric)
        .def("blade_name", &Signature::blade_name)
        .def("__repr__", [](const Signature& sig) {
            std::ostringstream stream;
            stream << "Signature(n=" << sig.n() << ", metric=[";
            for (int i = 0; i < sig.n(); ++i) {
                if (i) stream << ", ";
                stream << sig.metric(i);
            }
            stream << "])";
            return stream.str();
        });

    py::class_<Multivector>(m, "Multivector")
        .def(py::init<Signature>())
        .def(py::init<Signature, std::vector<double>>())
        .def_static("zero", &Multivector::zero)
        .def_static("scalar", &Multivector::scalar)
        .def_static("blade", &Multivector::blade)
        .def_static("vector", [](const Signature& sig, const std::vector<double>& coords) {
            return Multivector::vector(sig, coords);
        })
        .def_property_readonly("signature", &Multivector::signature)
        .def_property_readonly("size", &Multivector::size)
        .def("get", &Multivector::get)
        .def("scalar_part", &Multivector::scalar_part)
        .def("coefficients", &coefficients_numpy)
        .def("to_numpy", &coefficients_numpy)
        .def("add", py::overload_cast<const Multivector&>(&Multivector::add, py::const_))
        .def("add", py::overload_cast<double>(&Multivector::add, py::const_))
        .def("sub", py::overload_cast<const Multivector&>(&Multivector::sub, py::const_))
        .def("sub", py::overload_cast<double>(&Multivector::sub, py::const_))
        .def("scale", &Multivector::scale)
        .def("gp", &Multivector::gp)
        .def("op", &Multivector::op)
        .def("ip", &Multivector::ip)
        .def("reverse", &Multivector::reverse)
        .def("norm", &Multivector::norm)
        .def("unit", &Multivector::unit)
        .def("inverse", &Multivector::inverse)
        .def("sandwich", &Multivector::sandwich)
        .def("grade", &Multivector::grade)
        .def("__add__", py::overload_cast<const Multivector&>(&Multivector::add, py::const_))
        .def("__sub__", py::overload_cast<const Multivector&>(&Multivector::sub, py::const_))
        .def("__mul__", py::overload_cast<const Multivector&>(&Multivector::mul, py::const_))
        .def("__repr__", &repr_multivector);

    m.def("e3_vec", [](double x, double y, double z) { return E3::vec(x, y, z); });
    m.def("e3_bivector", &E3::bivector);
    m.def("e3_rotor", &E3::rotor);
    m.def("e3_rotate", &E3::rotate);
    m.def("e3_coords", [](const Multivector& value) { return to_numpy_3(E3::coords(value)); });

    m.def("p3_point", &P3::point);
    m.def("p3_point_coords", [](const Multivector& value) {
        return to_numpy_3(P3::point_coords(value));
    });
    m.def("p3_plane", &P3::plane);
    m.def("p3_plane_coords", [](const Multivector& value) {
        return to_numpy_4(P3::plane_coords(value));
    });
    m.def("p3_translator", &P3::translator);
    m.def("p3_rotor", &P3::rotor);
    m.def("p3_motor", &P3::motor);
    m.def("p3_apply", &P3::apply);
    m.def("p3_line", &P3::line);
    m.def("p3_line_direction", [](const Multivector& value) {
        return to_numpy_3(P3::line_direction(value));
    });
    m.def("p3_line_moment", [](const Multivector& value) {
        return to_numpy_3(P3::line_moment(value));
    });
    m.def("p3_line_anchor", [](const Multivector& value) {
        return to_numpy_3(P3::line_anchor(value));
    });
    m.def("p3_closest_point_on_line", &P3::closest_point_on_line);
    m.def("p3_distance_point_line", &P3::distance_point_line);
    m.def("p3_signed_distance_point_plane", &P3::signed_distance_point_plane);
    m.def("p3_distance_point_plane", &P3::distance_point_plane);
    m.def("p3_closest_point_on_plane", &P3::closest_point_on_plane);
    m.def("p3_meet_planes", &P3::meet_planes);
    m.def("p3_meet_plane_line", &P3::meet_plane_line);

    m.def("e3_rotate_many", &rotate_many, py::arg("vectors"), py::arg("rotor"),
          "Rotate an (N, 3) float64 array with one rotor, returning (N, 3).");
}
