/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "PythonUI.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Utils/Math/Vector.h"
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(std::vector<Falcor::ref<Falcor::python_ui::Widget>>);

namespace Falcor
{
namespace python_ui
{

/// Scoped push/pop of ImGui ID.
class ScopedID
{
public:
    ScopedID(void* id) { ImGui::PushID(id); }
    ~ScopedID() { ImGui::PopID(); }
};

/// Scoped begin/end for disabling ImGUI widgets.
class ScopedDisable
{
public:
    ScopedDisable(bool disabled) : m_disabled(disabled)
    {
        if (disabled)
            ImGui::BeginDisabled();
    }
    ~ScopedDisable()
    {
        if (m_disabled)
            ImGui::EndDisabled();
    }

private:
    bool m_disabled;
};

class Window : public Widget
{
    FALCOR_OBJECT(Window)
public:
    Window(Widget* parent, std::string_view title = "", float2 position = float2(10.f, 10.f), float2 size = float2(400.f, 400.f))
        : Widget(parent), m_title(title), m_position(position), m_size(size)
    {}

    const std::string& get_title() const { return m_title; }
    void set_title(std::string_view title) { m_title = title; }

    float2 get_position() const { return m_position; }
    void set_position(const float2& position)
    {
        m_position = position;
        m_set_position = true;
    }

    float2 get_size() const { return m_size; }
    void set_size(const float2& size)
    {
        m_size = size;
        m_set_size = true;
    }

    void show() { set_visible(true); }
    void close() { set_visible(false); }

    virtual void render() override
    {
        if (!m_visible)
            return;

        if (m_set_position)
        {
            ImGui::SetNextWindowPos(ImVec2(m_position.x, m_position.y));
            m_set_position = false;
        }
        if (m_set_size)
        {
            ImGui::SetNextWindowSize(ImVec2(m_size.x, m_size.y));
            m_set_size = false;
        }

        ScopedID id(this);
        if (ImGui::Begin(m_title.c_str(), &m_visible))
        {
            auto pos = ImGui::GetWindowPos();
            m_position = float2(pos.x, pos.y);
            auto size = ImGui::GetWindowSize();
            m_size = float2(size.x, size.y);

            ImGui::PushItemWidth(300);
            Widget::render();
            ImGui::PopItemWidth();
        }
        ImGui::End();
    }

private:
    std::string m_title;
    float2 m_position;
    float2 m_size;
    bool m_set_position{true};
    bool m_set_size{true};
};

class Group : public Widget
{
    FALCOR_OBJECT(Group)
public:
    Group(Widget* parent, std::string_view label = "") : Widget(parent), m_label(label) {}

    const std::string& get_label() const { return m_label; }
    void set_label(std::string_view label) { m_label = label; }

    virtual void render() override
    {
        // Check if this is a nested group
        bool nested = false;
        for (Widget* p = get_parent(); p != nullptr; p = p->get_parent())
            if (dynamic_cast<Group*>(p) != nullptr)
                nested = true;

        ScopedID id(this);
        ScopedDisable disable(!m_enabled);

        if (nested ? ImGui::TreeNodeEx(m_label.c_str(), ImGuiTreeNodeFlags_DefaultOpen)
                   : ImGui::CollapsingHeader(m_label.c_str(), ImGuiTreeNodeFlags_DefaultOpen))
        {
            Widget::render();
            if (nested)
                ImGui::TreePop();
        }
    }

private:
    std::string m_label;
};

class Text : public Widget
{
    FALCOR_OBJECT(Text)
public:
    Text(Widget* parent, std::string_view text = "") : Widget(parent), m_text(text) {}

    const std::string& get_text() const { return m_text; }
    void set_text(std::string_view text) { m_text = text; }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        ImGui::TextUnformatted(m_text.c_str());
    }

private:
    std::string m_text;
};

class ProgressBar : public Widget
{
    FALCOR_OBJECT(ProgressBar)
public:
    ProgressBar(Widget* parent, float fraction = 0.f) : Widget(parent), m_fraction(fraction) {}

    float get_fraction() const { return m_fraction; }
    void set_fraction(float fraction) { m_fraction = fraction; }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        ImGui::ProgressBar(m_fraction);
    }

private:
    float m_fraction;
};

class Button : public Widget
{
    FALCOR_OBJECT(Button)
public:
    using Callback = std::function<void()>;

    Button(Widget* parent, std::string_view label = "", Callback callback = {}) : Widget(parent), m_label(label), m_callback(callback) {}

    const std::string& get_label() const { return m_label; }
    void set_label(std::string_view label) { m_label = label; }

    Callback get_callback() const { return m_callback; }
    void set_callback(Callback callback) { m_callback = callback; }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::Button(m_label.c_str()))
        {
            if (m_callback)
                m_callback();
        }
    }

private:
    std::string m_label;
    Callback m_callback;
};

class Property : public Widget
{
    FALCOR_OBJECT(Property)
public:
    using ChangeCallback = std::function<void()>;

    Property(Widget* parent, std::string_view label, ChangeCallback change_callback)
        : Widget(parent), m_label(label), m_change_callback(change_callback)
    {}

    const std::string& get_label() const { return m_label; }
    void set_label(std::string_view label) { m_label = label; }

    ChangeCallback get_change_callback() const { return m_change_callback; }
    void set_change_callback(ChangeCallback change_callback) { m_change_callback = change_callback; }

protected:
    std::string m_label;
    ChangeCallback m_change_callback;
};

class Checkbox : public Property
{
    FALCOR_OBJECT(Checkbox)
public:
    Checkbox(Widget* parent, std::string_view label = "", ChangeCallback change_callback = {}, bool value = false)
        : Property(parent, label, change_callback), m_value(value)
    {}

    bool get_value() const { return m_value; }
    void set_value(bool value) { m_value = value; }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::Checkbox(m_label.c_str(), &m_value))
        {
            if (m_change_callback)
                m_change_callback();
        }
    }

private:
    bool m_value;
};

class Combobox : public Property
{
    FALCOR_OBJECT(Combobox)
public:
    Combobox(
        Widget* parent,
        std::string_view label = "",
        ChangeCallback change_callback = {},
        std::vector<std::string> items = {},
        int value = 0
    )
        : Property(parent, label, change_callback), m_items(items), m_value(value)
    {}

    const std::vector<std::string>& get_items() const { return m_items; }
    void set_items(const std::vector<std::string>& items) { m_items = items; }

    int get_value() const { return m_value; }
    void set_value(int value) { m_value = std::clamp(value, 0, (int)m_items.size()); }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        if (ImGui::Combo(
                m_label.c_str(),
                &m_value,
                [](void* data, int idx, const char** out_text) -> bool
                {
                    auto& items = *reinterpret_cast<std::vector<std::string>*>(data);
                    if (idx < 0 || idx >= items.size())
                        return false;
                    *out_text = items[idx].c_str();
                    return true;
                },
                &m_items,
                (int)m_items.size()
            ))
        {
            if (m_change_callback)
                m_change_callback();
        }
    }

private:
    std::vector<std::string> m_items;
    int m_value;
};

enum class SliderFlags
{
    None = ImGuiSliderFlags_None,
    AlwaysClamp = ImGuiSliderFlags_AlwaysClamp,
    Logarithmic = ImGuiSliderFlags_Logarithmic,
    NoRoundToFormat = ImGuiSliderFlags_NoRoundToFormat,
    NoInput = ImGuiSliderFlags_NoInput,
};

template<typename T>
struct SliderTraits
{};

// clang-format off
template<> struct SliderTraits<float> { using type = float; using scalar_type = float; static constexpr bool is_float = true; static constexpr int N = 1; static constexpr const char* default_format = "%.3f"; };
template<> struct SliderTraits<float2> { using type = float2; using scalar_type = float; static constexpr bool is_float = true; static constexpr int N = 2; static constexpr const char* default_format = "%.3f"; };
template<> struct SliderTraits<float3> { using type = float3; using scalar_type = float; static constexpr bool is_float = true; static constexpr int N = 3; static constexpr const char* default_format = "%.3f"; };
template<> struct SliderTraits<float4> { using type = float4; using scalar_type = float; static constexpr bool is_float = true; static constexpr int N = 4; static constexpr const char* default_format = "%.3f"; };
template<> struct SliderTraits<int> { using type = int; using scalar_type = int; static constexpr bool is_float = false; static constexpr int N = 1; static constexpr const char* default_format = "%d"; };
template<> struct SliderTraits<int2> { using type = int2; using scalar_type = int; static constexpr bool is_float = false; static constexpr int N = 2; static constexpr const char* default_format = "%d"; };
template<> struct SliderTraits<int3> { using type = int3; using scalar_type = int; static constexpr bool is_float = false; static constexpr int N = 3; static constexpr const char* default_format = "%d"; };
template<> struct SliderTraits<int4> { using type = int4; using scalar_type = int; static constexpr bool is_float = false; static constexpr int N = 4; static constexpr const char* default_format = "%d"; };
// clang-format on

template<typename T>
class Drag : public Property
{
    FALCOR_OBJECT(Drag)
public:
    using traits = SliderTraits<T>;
    using type = typename traits::type;
    using scalar_type = typename traits::scalar_type;
    static constexpr bool is_float = traits::is_float;
    static constexpr int N = traits::N;
    static constexpr const char* default_format = traits::default_format;

    Drag(
        Widget* parent,
        std::string_view label = "",
        ChangeCallback change_callback = {},
        type value = type(0),
        float speed = 1.f,
        scalar_type min = scalar_type(0),
        scalar_type max = scalar_type(0),
        std::string_view format = default_format,
        SliderFlags flags = SliderFlags::None
    )
        : Property(parent, label, change_callback), m_value(value), m_speed(speed), m_min(min), m_max(max), m_format(format), m_flags(flags)
    {}

    type get_value() const { return m_value; }
    void set_value(type value) { m_value = value; }

    scalar_type get_speed() const { return m_speed; }
    void set_speed(scalar_type speed) { m_speed = speed; }

    scalar_type get_min() const { return m_min; }
    void set_min(scalar_type min) { m_min = min; }

    scalar_type get_max() const { return m_max; }
    void set_max(scalar_type max) { m_max = max; }

    const std::string& get_format() const { return m_format; }
    void set_format(std::string_view format) { m_format = format; }

    SliderFlags get_flags() const { return m_flags; }
    void set_flags(SliderFlags flags) { m_flags = flags; }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        bool changed = false;
        if constexpr (is_float == true)
        {
            if constexpr (N == 1)
                changed = ImGui::DragFloat(m_label.c_str(), &m_value, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 2)
                changed =
                    ImGui::DragFloat2(m_label.c_str(), &m_value.x, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 3)
                changed =
                    ImGui::DragFloat3(m_label.c_str(), &m_value.x, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 4)
                changed =
                    ImGui::DragFloat4(m_label.c_str(), &m_value.x, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
        }
        else
        {
            if constexpr (N == 1)
                changed = ImGui::DragInt(m_label.c_str(), &m_value, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 2)
                changed = ImGui::DragInt2(m_label.c_str(), &m_value.x, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 3)
                changed = ImGui::DragInt3(m_label.c_str(), &m_value.x, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 4)
                changed = ImGui::DragInt4(m_label.c_str(), &m_value.x, m_speed, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
        }

        if (changed && m_change_callback)
            m_change_callback();
    }

private:
    type m_value;
    float m_speed;
    scalar_type m_min;
    scalar_type m_max;
    std::string m_format;
    SliderFlags m_flags;
};

using DragFloat = Drag<float>;
using DragFloat2 = Drag<float2>;
using DragFloat3 = Drag<float3>;
using DragFloat4 = Drag<float4>;
using DragInt = Drag<int>;
using DragInt2 = Drag<int2>;
using DragInt3 = Drag<int3>;
using DragInt4 = Drag<int4>;

template<typename T>
class Slider : public Property
{
    FALCOR_OBJECT(Slider)
public:
    using traits = SliderTraits<T>;
    using type = typename traits::type;
    using scalar_type = typename traits::scalar_type;
    static constexpr bool is_float = traits::is_float;
    static constexpr int N = traits::N;
    static constexpr const char* default_format = traits::default_format;

    Slider(
        Widget* parent,
        std::string_view label = "",
        ChangeCallback change_callback = {},
        type value = type(0),
        scalar_type min = scalar_type(0),
        scalar_type max = scalar_type(0),
        std::string_view format = default_format,
        SliderFlags flags = SliderFlags::None
    )
        : Property(parent, label, change_callback), m_value(value), m_min(min), m_max(max), m_format(format), m_flags(flags)
    {}

    type get_value() const { return m_value; }
    void set_value(type value) { m_value = value; }

    scalar_type get_min() const { return m_min; }
    void set_min(scalar_type min) { m_min = min; }

    scalar_type get_max() const { return m_max; }
    void set_max(scalar_type max) { m_max = max; }

    const std::string& get_format() const { return m_format; }
    void set_format(std::string_view format) { m_format = format; }

    SliderFlags get_flags() const { return m_flags; }
    void set_flags(SliderFlags flags) { m_flags = flags; }

    virtual void render() override
    {
        ScopedID id(this);
        ScopedDisable disable(!m_enabled);
        bool changed = false;
        if constexpr (is_float == true)
        {
            if constexpr (N == 1)
                changed = ImGui::SliderFloat(m_label.c_str(), &m_value, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 2)
                changed = ImGui::SliderFloat2(m_label.c_str(), &m_value.x, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 3)
                changed = ImGui::SliderFloat3(m_label.c_str(), &m_value.x, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 4)
                changed = ImGui::SliderFloat4(m_label.c_str(), &m_value.x, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
        }
        else
        {
            if constexpr (N == 1)
                changed = ImGui::SliderInt(m_label.c_str(), &m_value, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 2)
                changed = ImGui::SliderInt2(m_label.c_str(), &m_value.x, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 3)
                changed = ImGui::SliderInt3(m_label.c_str(), &m_value.x, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
            if constexpr (N == 4)
                changed = ImGui::SliderInt4(m_label.c_str(), &m_value.x, m_min, m_max, m_format.c_str(), (ImGuiSliderFlags)m_flags);
        }

        if (changed && m_change_callback)
            m_change_callback();
    }

private:
    type m_value;
    scalar_type m_min;
    scalar_type m_max;
    std::string m_format;
    SliderFlags m_flags;
};

using SliderFloat = Slider<float>;
using SliderFloat2 = Slider<float2>;
using SliderFloat3 = Slider<float3>;
using SliderFloat4 = Slider<float4>;
using SliderInt = Slider<int>;
using SliderInt2 = Slider<int2>;
using SliderInt3 = Slider<int3>;
using SliderInt4 = Slider<int4>;

template<typename T>
static void bind_drag(pybind11::module_ m, const char* name)
{
    using namespace pybind11::literals;

    pybind11::class_<T, Property, ref<T>> drag(m, name);
    drag.def(
        pybind11::init<
            Widget*,
            std::string_view,
            typename T::ChangeCallback,
            typename T::type,
            float,
            typename T::scalar_type,
            typename T::scalar_type,
            std::string_view,
            SliderFlags>(),
        "parent"_a,
        "label"_a = "",
        "change_callback"_a = typename T::ChangeCallback{},
        "value"_a = typename T::type(0),
        "speed"_a = 1.f,
        "min"_a = typename T::scalar_type(0),
        "max"_a = typename T::scalar_type(0),
        "format"_a = T::default_format,
        "flags"_a = SliderFlags::None
    );
    drag.def_property("value", &T::get_value, &T::set_value);
    drag.def_property("speed", &T::get_speed, &T::set_speed);
    drag.def_property("min", &T::get_min, &T::set_min);
    drag.def_property("max", &T::get_max, &T::set_max);
    drag.def_property("format", &T::get_format, &T::set_format);
    drag.def_property("flags", &T::get_flags, &T::set_flags);
}

template<typename T>
static void bind_slider(pybind11::module_ m, const char* name)
{
    using namespace pybind11::literals;

    pybind11::class_<T, Property, ref<T>> drag(m, name);
    drag.def(
        pybind11::init<
            Widget*,
            std::string_view,
            typename T::ChangeCallback,
            typename T::type,
            typename T::scalar_type,
            typename T::scalar_type,
            std::string_view,
            SliderFlags>(),
        "parent"_a,
        "label"_a = "",
        "change_callback"_a = typename T::ChangeCallback{},
        "value"_a = typename T::type(0),
        "min"_a = typename T::scalar_type(0),
        "max"_a = typename T::scalar_type(0),
        "format"_a = T::default_format,
        "flags"_a = SliderFlags::None
    );
    drag.def_property("value", &T::get_value, &T::set_value);
    drag.def_property("min", &T::get_min, &T::set_min);
    drag.def_property("max", &T::get_max, &T::set_max);
    drag.def_property("format", &T::get_format, &T::set_format);
    drag.def_property("flags", &T::get_flags, &T::set_flags);
}

FALCOR_SCRIPT_BINDING(python_ui)
{
    using namespace pybind11::literals;

    pybind11::module_ ui = m.def_submodule("ui");

    pybind11::class_<Widget, ref<Widget>> widget(ui, "Widget");

    pybind11::bind_vector<std::vector<ref<Widget>>>(ui, "WidgetVector");

    widget.def_property("parent", (Widget * (Widget::*)(void)) & Widget::get_parent, &Widget::set_parent);
    widget.def_property_readonly("children", &Widget::get_children);
    widget.def_property("visible", &Widget::get_visible, &Widget::set_visible);
    widget.def_property("enabled", &Widget::get_enabled, &Widget::set_enabled);

    pybind11::class_<Screen, Widget, ref<Screen>> screen(ui, "Screen");

    pybind11::class_<Window, Widget, ref<Window>> window(ui, "Window");
    window.def(
        pybind11::init<Widget*, std::string_view, float2, float2>(),
        "parent"_a,
        "title"_a = "",
        "position"_a = float2(10.f, 10.f),
        "size"_a = float2(400.f, 400.f)
    );
    window.def("show", &Window::show);
    window.def("close", &Window::close);
    window.def_property("title", &Window::get_title, &Window::set_title);
    window.def_property("position", &Window::get_position, &Window::set_position);
    window.def_property("size", &Window::get_size, &Window::set_size);

    pybind11::class_<Group, Widget, ref<Group>> group(ui, "Group");
    group.def(pybind11::init<Widget*, std::string_view>(), "parent"_a, "label"_a = "");
    group.def_property("label", &Group::get_label, &Group::set_label);

    pybind11::class_<Text, Widget, ref<Text>> text(ui, "Text");
    text.def(pybind11::init<Widget*, std::string_view>(), "parent"_a, "text"_a = "");
    text.def_property("text", &Text::get_text, &Text::set_text);

    pybind11::class_<ProgressBar, Widget, ref<ProgressBar>> progress_bar(ui, "ProgressBar");
    progress_bar.def(pybind11::init<Widget*, float>(), "parent"_a, "fraction"_a = 0.f);
    progress_bar.def_property("fraction", &ProgressBar::get_fraction, &ProgressBar::set_fraction);

    pybind11::class_<Button, Widget, ref<Button>> button(ui, "Button");
    button.def(
        pybind11::init<Widget*, std::string_view, Button::Callback>(), "parent"_a, "label"_a = "", "callback"_a = Button::Callback{}
    );
    button.def_property("label", &Button::get_label, &Button::set_label);
    button.def_property("callback", &Button::get_callback, &Button::set_callback);

    pybind11::class_<Property, Widget, ref<Property>> property(ui, "Property");
    property.def_property("label", &Property::get_label, &Property::set_label);
    property.def_property("change_callback", &Property::get_change_callback, &Property::set_change_callback);

    pybind11::class_<Checkbox, Property, ref<Checkbox>> checkbox(ui, "Checkbox");
    checkbox.def(
        pybind11::init<Widget*, std::string_view, Checkbox::ChangeCallback, bool>(),
        "parent"_a,
        "label"_a = "",
        "change_callback"_a = Checkbox::ChangeCallback{},
        "value"_a = false
    );
    checkbox.def_property("value", &Checkbox::get_value, &Checkbox::set_value);

    pybind11::class_<Combobox, Property, ref<Combobox>> combobox(ui, "Combobox");
    combobox.def(
        pybind11::init<Widget*, std::string_view, Combobox::ChangeCallback, std::vector<std::string>, int>(),
        "parent"_a,
        "label"_a = "",
        "change_callback"_a = Combobox::ChangeCallback{},
        "items"_a = std::vector<std::string>{},
        "value"_a = 0
    );
    combobox.def_property("items", &Combobox::get_items, &Combobox::set_items);
    combobox.def_property("value", &Combobox::get_value, &Combobox::set_value);

    pybind11::enum_<SliderFlags> slider_flags(ui, "SliderFlags");
    slider_flags.value("None_", SliderFlags::None);
    slider_flags.value("AlwaysClamp", SliderFlags::AlwaysClamp);
    slider_flags.value("Logarithmic", SliderFlags::Logarithmic);
    slider_flags.value("NoRoundToFormat", SliderFlags::NoRoundToFormat);
    slider_flags.value("NoInput", SliderFlags::NoInput);

    bind_drag<DragFloat>(ui, "DragFloat");
    bind_drag<DragFloat2>(ui, "DragFloat2");
    bind_drag<DragFloat3>(ui, "DragFloat3");
    bind_drag<DragFloat4>(ui, "DragFloat4");
    bind_drag<DragInt>(ui, "DragInt");
    bind_drag<DragInt2>(ui, "DragInt2");
    bind_drag<DragInt3>(ui, "DragInt3");
    bind_drag<DragInt4>(ui, "DragInt4");

    bind_slider<SliderFloat>(ui, "SliderFloat");
    bind_slider<SliderFloat2>(ui, "SliderFloat2");
    bind_slider<SliderFloat3>(ui, "SliderFloat3");
    bind_slider<SliderFloat4>(ui, "SliderFloat4");
    bind_slider<SliderInt>(ui, "SliderInt");
    bind_slider<SliderInt2>(ui, "SliderInt2");
    bind_slider<SliderInt3>(ui, "SliderInt3");
    bind_slider<SliderInt4>(ui, "SliderInt4");
}

} // namespace python_ui

} // namespace Falcor
