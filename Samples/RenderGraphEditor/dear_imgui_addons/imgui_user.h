// requires:
// defining IMGUI_INCLUDE_IMGUI_USER_H and IMGUI_INCLUDE_IMGUI_USER_INL
// at the project level

#pragma once
#ifndef IMGUI_USER_ADDONS_H_
#define IMGUI_USER_ADDONS_H_

#ifndef IMGUI_INCLUDE_IMGUI_USER_INL
//#warning IMGUI_INCLUDE_IMGUI_USER_INL should be defined
#define IMGUI_INCLUDE_IMGUI_USER_INL
#endif //IMGUI_INCLUDE_IMGUI_USER_INL

#ifndef IMGUI_FORCE_INLINE
#	ifdef _MSC_VER
#		define IMGUI_FORCE_INLINE __forceinline
#	elif (defined(__clang__) || defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__))
#		define IMGUI_FORCE_INLINE inline __attribute__((__always_inline__))
#	else
#		define IMGUI_FORCE_INLINE inline
#	endif
#endif//IMGUI_FORCE_INLINE

#ifndef IMGUI_NO_INLINE
#	ifdef _MSC_VER
#		define IMGUI_NO_INLINE __declspec((noinline))
#	elif (defined(__clang__) || defined(__GNUC__) || defined(__MINGW32__) || defined(__MINGW64__))
#		define IMGUI_NO_INLINE __attribute__((__noinline__))
#	else
#		define IMGUI_NO_INLINE
#	endif
#endif//IMGUI_NO_INLINE

#ifdef NO_IMGUIADDONS	// Mispell fix
#	undef NO_IMGUI_ADDONS
#	define NO_IMGUI_ADDONS
#endif //NO_IMGUIADDONS

#ifdef NO_IMGUI_ADDONS  // This definition turns all "normal" addons into "yes_addons"
#   if (!defined(YES_IMGUISTYLESERIALIZER) && !defined(NO_IMGUISTYLESERIALIZER))
#       define NO_IMGUISTYLESERIALIZER
#   endif //YES_IMGUISTYLESERIALIZER
#   if (!defined(YES_IMGUIFILESYSTEM) && !defined(NO_IMGUIFILESYSTEM))
#       define NO_IMGUIFILESYSTEM
#   endif //YES_IMGUIFILESYSTEM
#   if (!defined(YES_IMGUIDATECHOOSER) && !defined(NO_IMGUIDATECHOOSER))
#       define NO_IMGUIDATECHOOSER
#   endif //YES_IMGUIDATECHOOSER
#   if (!defined(YES_IMGUILISTVIEW) && !defined(NO_IMGUILISTVIEW))
#       define NO_IMGUILISTVIEW
#   endif //YES_IMGUILISTVIEW
#   if (!defined(YES_IMGUITOOLBAR) && !defined(NO_IMGUITOOLBAR))
#       define NO_IMGUITOOLBAR
#   endif //YES_IMGUITOOLBAR
#   if (!defined(YES_IMGUIPANELMANAGER) && !defined(NO_IMGUIPANELMANAGER))
#       define NO_IMGUIPANELMANAGER
#   endif //YES_IMGUIPANELMANAGER
#   if (!defined(YES_IMGUITABWINDOW) && !defined(NO_IMGUITABWINDOW))
#       define NO_IMGUITABWINDOW
#   endif //YES_IMGUITABWINDOW
#   if (!defined(YES_IMGUIDOCK) && !defined(NO_IMGUIDOCK))
#       define NO_IMGUIDOCK
#   endif //YES_IMGUIDOCK
#   if (!defined(YES_IMGUINODEGRAPHEDITOR) && !defined(NO_IMGUINODEGRAPHEDITOR))
#       define NO_IMGUINODEGRAPHEDITOR
#   endif //YES_IMGUINODEGRAPHEDITOR
#   if (!defined(YES_IMGUICODEEDITOR) && !defined(NO_IMGUICODEEDITOR))
#       define NO_IMGUICODEEDITOR
#   endif //YES_IMGUICODEEDITOR
#   if (!defined(YES_IMGUISTRING) && !defined(NO_IMGUISTRING))
#       define NO_IMGUISTRING
#   endif //YES_IMGUISTRING
#   if (!defined(YES_IMGUIHELPER) && !defined(NO_IMGUIHELPER))
#       define NO_IMGUIHELPER
#   endif //YES_IMGUIHELPER
#   if (!defined(YES_IMGUIVARIOUSCONTROLS) && !defined(NO_IMGUIVARIOUSCONTROLS))
#       define NO_IMGUIVARIOUSCONTROLS
#   endif //YES_IMGUIVARIOUSCONTROLS
#endif // NO_IMGUI_ADDONS

// Defining a custom placement new() with a dummy parameter allows us to bypass including <new> which on some platforms complains when user has disabled exceptions.
// (This is necessary because the original IM_PLACEMENT_NEW macro is not exposed by imgui.h, but only by imgui_internal.h)
#ifndef IMIMPL_HAS_PLACEMENT_NEW
#define IMIMPL_HAS_PLACEMENT_NEW
struct ImImplPlacementNewDummy {};
inline void* operator new(size_t, ImImplPlacementNewDummy, void* ptr) { return ptr; }
inline void operator delete(void*, ImImplPlacementNewDummy, void*) {}
#define IMIMPL_PLACEMENT_NEW(_PTR)  new(ImImplPlacementNewDummy() ,_PTR)
#endif //IMIMPL_HAS_PLACEMENT_NEW

#ifdef IMGUI_USE_MINIZIP	// requires linking to library -lZlib
#   ifndef IMGUI_USE_ZLIB
#   define IMGUI_USE_ZLIB	// requires linking to library -lZlib
#   endif //IMGUI_USE_ZLIB
#endif //IMGUI_USE_MINIZIP

// We add these yes_addons before imguibindings.h
#ifdef YES_IMGUIADDONS_ALL
#	ifndef NO_IMGUIBZ2
#		undef YES_IMGUIBZ2
#		define YES_IMGUIBZ2
#	endif //NO_IMGUIBZ2
#	ifndef NO_IMGUISTRINGIFIER
#		undef YES_IMGUISTRINGIFIER
#		define YES_IMGUISTRINGIFIER
#	endif //NO_IMGUISTRINGIFIER
#endif // YES_IMGUIADDONS_ALL

#ifdef YES_IMGUIBZ2
#include "./imguiyesaddons/imguibz2.h"
#endif //YES_IMGUIBZ2
#ifdef YES_IMGUISTRINGIFIER
#include "./imguiyesaddons/imguistringifier.h"
#endif //YES_IMGUISTRINGIFIER

#ifdef __EMSCRIPTEN__
#	ifdef NO_IMGUIEMSCRIPTEN
#		warning NO_IMGUIEMSCRIPTEN is deprecated (it is defined by default). YES_IMGUIEMSCRIPTENPERSISTENTFOLDER can be used to enable the yes_addon.
#	endif //NO_IMGUIEMSCRIPTEN
#   ifdef YES_IMGUIEMSCRIPTENPERSISTENTFOLDER
#       include "./imguiyesaddons/imguiemscriptenpersistentfolder.h"
#   endif //YES_IMGUIEMSCRIPTENPERSISTENTFOLDER
#else //__EMSCRIPTEN__
#   undef YES_IMGUIEMSCRIPTENPERSISTENTFOLDER
#endif //__EMSCRIPTEN__

#ifndef NO_IMGUISTRING
#include "./imguistring/imguistring.h"
#endif //NO_IMGUISTRING
#ifndef NO_IMGUIHELPER
#include "./imguihelper/imguihelper.h"
#endif //NO_IMGUIHELPER
#ifndef NO_IMGUITABWINDOW
#include "./imguitabwindow/imguitabwindow.h"
#endif //NO_IMGUITABWINDOW

#ifdef YES_IMGUISOLOUD_ALL
#   undef YES_IMGUISOLOUD
#   define YES_IMGUISOLOUD
#endif //YES_IMGUISOLOUD_ALL

#if (defined(YES_IMGUISOLOUD) || defined(YES_IMGUIADDONS_ALL))
// If no SoLoud backend is defined, define one. We must do it here to integrate with ImGui bindings better.
// Available bindings beside SDL: WITH_PORTAUDIO  WITH_OPENAL WITH_XAUDIO2 WITH_WINMM WITH_WASAPIWITH_OSS WITH_ALSA (all untested)
#   if (!defined(WITH_SDL) && !defined(WITH_SDL_STATIC) && !defined(WITH_SDL2) && !defined(WITH_SDL2_STATIC) && !defined(WITH_PORTAUDIO)  && !defined(WITH_OPENAL) && !defined(WITH_XAUDIO2) && !defined(WITH_WINMM)  && !defined(WITH_WASAPI) && !defined(WITH_OSS) && !defined(WITH_ALSA) && !defined(WITH_NULLDRIVER))
#       ifdef IMGUI_USE_SDL2_BINDING
#           define WITH_SDL2_STATIC         // So in our SDL2 binding we can force initialization of SDL_AUDIO
#       else //IMGUI_USE_SDL2_BINDING
#           if (defined(_WIN32) || defined(_WIN64))
#               define WITH_WINMM
//#           elif (defined(__linux__))
//#               define WITH_ALSA        // needs the alsa lib (PKGCONFIG += alsa in QtCreator)
#           else // (defined(_WIN32) || defined(_WIN64))
#               define WITH_OPENAL          // Or maybe some other specific for Linux... nope: this can link dynamically
#           endif // (defined(_WIN32) || defined(_WIN64))
#       endif //IMGUI_USE_SDL2_BINDING
#   endif // NO_SOLOUD_BINDING
#endif //YES_IMGUISOLOUD


#undef IMGUI_USE_AUTO_BINDING
#undef IMGUI_USE_AUTO_BINDING_OPENGL
#undef IMGUI_USE_AUTO_BINDING_DIRECT3D
#undef IMGUI_USE_AUTO_BINDING_WINDOWS
#if (defined(IMGUI_USE_GLUT_BINDING) || defined(IMGUI_USE_SDL2_BINDING) || defined(IMGUI_USE_GLFW_BINDING) || defined(IMGUI_USE_WINAPI_BINDING) || defined(IMGUI_USE_DIRECT3D9_BINDING))
#	define IMGUI_USE_AUTO_BINDING
#   if (!defined(IMGUI_USE_DIRECT3D9_BINDING))
#   define IMGUI_USE_AUTO_BINDING_OPENGL
#   else //(!defined(IMGUI_USE_DIRECT3D9_BINDING))
#   define IMGUI_USE_AUTO_BINDING_DIRECT3D
#   endif //(!defined(IMGUI_USE_DIRECT3D9_BINDING))
#   if (defined(IMGUI_USE_WINAPI_BINDING) || defined(IMGUI_USE_DIRECT3D9_BINDING))
#   define IMGUI_USE_AUTO_BINDING_WINDOWS
#   endif //
# 	include "./imguibindings/imguibindings.h"
#endif //IMGUI_USE_AUTO_BINDING

#ifndef NO_IMGUILISTVIEW
#include "./imguilistview/imguilistview.h"
#endif //NO_IMGUILISTVIEW
#ifndef NO_IMGUIFILESYSTEM
#include "./imguifilesystem/imguifilesystem.h"
#endif //NO_IMGUIFILESYSTEM
#ifndef NO_IMGUITOOLBAR
#include "./imguitoolbar/imguitoolbar.h"
#endif //NO_IMGUITOOLBAR
#ifndef NO_IMGUIPANELMANAGER
#include "./imguipanelmanager/imguipanelmanager.h"
#endif //NO_IMGUIPANELMANAGER
#ifndef NO_IMGUIVARIOUSCONTROLS
#include "./imguivariouscontrols/imguivariouscontrols.h"
#endif //NO_IMGUIVARIOUSCONTROLS
#ifndef NO_IMGUISTYLESERIALIZER
#include "./imguistyleserializer/imguistyleserializer.h"
#endif //NO_IMGUISTYLESERIALIZER
#ifndef NO_IMGUIDATECHOOSER
#include "./imguidatechooser/imguidatechooser.h"
#endif //NO_IMGUIDATECHOOSER
#ifndef NO_IMGUICODEEDITOR
#include "./imguicodeeditor/imguicodeeditor.h"
#endif //NO_IMGUICODEEDITOR
#ifndef NO_IMGUINODEGRAPHEDITOR
#include "./imguinodegrapheditor/imguinodegrapheditor.h"
#endif //NO_IMGUINODEGRAPHEDITOR
#ifndef NO_IMGUIDOCK
#include "./imguidock/imguidock.h"
#endif //NO_IMGUIDOCK

#ifdef YES_IMGUIADDONS_ALL
#	ifndef NO_IMGUIPDFVIEWER
#		undef YES_IMGUIPDFVIEWER
#		define YES_IMGUIPDFVIEWER
#	endif //NO_IMGUIPDFVIEWER
#	ifdef IMGUI_USE_AUTO_BINDING_OPENGL
#		ifndef NO_IMGUISDF
#			undef YES_IMGUISDF
#			define YES_IMGUISDF
#		endif //NO_IMGUISDF
#	endif //IMGUI_USE_AUTO_BINDING_OPENGL
#	ifndef NO_IMGUITINYFILEDIALOGS
#		undef YES_IMGUITINYFILEDIALOGS
#		define YES_IMGUITINYFILEDIALOGS
#	endif //NO_IMGUITINYFILEDIALOGS
#	if (!defined(NO_IMGUISQLITE3) && !defined(NO_IMGUISQLITE))
#		undef YES_IMGUISQLITE3
#		define YES_IMGUISQLITE3
#	endif //NO_IMGUISQLITE3
#	ifndef NO_IMGUIIMAGEEDITOR
#		undef YES_IMGUIIMAGEEDITOR
#		define YES_IMGUIIMAGEEDITOR
#	endif //NO_IMGUIIMAGEEDITOR
#	ifndef NO_IMGUISOLOUD
#		undef YES_IMGUISOLOUD
#		define YES_IMGUISOLOUD
#	endif //NO_IMGUISOLOUD
#	ifndef NO_IMGUIMINIGAMES
#		undef YES_IMGUIMINIGAMES
#		define YES_IMGUIMINIGAMES
#	endif //NO_IMGUIMINIGAMES
//#	ifndef NO_IMGUIFREETYPE	// We leave YES_IMGUIFREETYPE out
//#		undef YES_IMGUIFREETYPE
//#		define YES_IMGUIFREETYPE
//#	endif //NO_IMGUIFREETYPE
#endif //YES_IMGUIADDONS_ALL

#ifdef YES_IMGUIPDFVIEWER
#include "./imguiyesaddons/imguipdfviewer.h"
#endif //YES_IMGUIPDFVIEWER
#ifdef YES_IMGUISDF_MSDF_MODE
#undef YES_IMGUISDF
#define YES_IMGUISDF
#endif //YES_IMGUISDF_MSDF_MODE
#ifdef YES_IMGUISDF
#include "./imguiyesaddons/imguisdf.h"
#endif //YES_IMGUISDF
#ifdef YES_IMGUITINYFILEDIALOGS
#include "./imguiyesaddons/imguitinyfiledialogs.h"
#endif //YES_IMGUITINYFILEDIALOGS
#if (defined(YES_IMGUISQLITE3) || defined(YES_IMGUISQLITE))
#undef YES_IMGUISQLITE3
#define YES_IMGUISQLITE3
#undef YES_IMGUISQLITE
#define YES_IMGUISQLITE
#include "./imguiyesaddons/imguisqlite3.h"
#endif //YES_IMGUISQLITE3
#ifdef YES_IMGUIIMAGEEDITOR
#include "./imguiyesaddons/imguiimageeditor.h"
#endif //YES_IMGUIIMAGEEDITOR
#ifdef YES_IMGUIFREETYPE
#include "./imguiyesaddons/imguifreetype.h"
#endif //YES_IMGUIFREETYPE
#ifdef YES_IMGUISOLOUD
#include "./imguiyesaddons/imguisoloud.h" // Better leave it at the end...
#endif //YES_IMGUISOLOUD
#ifdef YES_IMGUIMINIGAMES
#include "./imguiyesaddons/imguiminigames.h" // ...but mini games could be enhanced by having sound...
#endif //YES_IMGUIMINIGAMES


#endif //IMGUI_USER_ADDONS_H_

