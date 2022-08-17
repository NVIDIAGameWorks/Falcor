/******/ (function(modules) { // webpackBootstrap
/******/ 	// The module cache
/******/ 	var installedModules = {};
/******/
/******/ 	// The require function
/******/ 	function __webpack_require__(moduleId) {
/******/
/******/ 		// Check if module is in cache
/******/ 		if(installedModules[moduleId]) {
/******/ 			return installedModules[moduleId].exports;
/******/ 		}
/******/ 		// Create a new module (and put it into the cache)
/******/ 		var module = installedModules[moduleId] = {
/******/ 			i: moduleId,
/******/ 			l: false,
/******/ 			exports: {}
/******/ 		};
/******/
/******/ 		// Execute the module function
/******/ 		modules[moduleId].call(module.exports, module, module.exports, __webpack_require__);
/******/
/******/ 		// Flag the module as loaded
/******/ 		module.l = true;
/******/
/******/ 		// Return the exports of the module
/******/ 		return module.exports;
/******/ 	}
/******/
/******/
/******/ 	// expose the modules object (__webpack_modules__)
/******/ 	__webpack_require__.m = modules;
/******/
/******/ 	// expose the module cache
/******/ 	__webpack_require__.c = installedModules;
/******/
/******/ 	// define getter function for harmony exports
/******/ 	__webpack_require__.d = function(exports, name, getter) {
/******/ 		if(!__webpack_require__.o(exports, name)) {
/******/ 			Object.defineProperty(exports, name, {
/******/ 				configurable: false,
/******/ 				enumerable: true,
/******/ 				get: getter
/******/ 			});
/******/ 		}
/******/ 	};
/******/
/******/ 	// getDefaultExport function for compatibility with non-harmony modules
/******/ 	__webpack_require__.n = function(module) {
/******/ 		var getter = module && module.__esModule ?
/******/ 			function getDefault() { return module['default']; } :
/******/ 			function getModuleExports() { return module; };
/******/ 		__webpack_require__.d(getter, 'a', getter);
/******/ 		return getter;
/******/ 	};
/******/
/******/ 	// Object.prototype.hasOwnProperty.call
/******/ 	__webpack_require__.o = function(object, property) { return Object.prototype.hasOwnProperty.call(object, property); };
/******/
/******/ 	// __webpack_public_path__
/******/ 	__webpack_require__.p = "./";
/******/
/******/ 	// Load entry module and return exports
/******/ 	return __webpack_require__(__webpack_require__.s = 0);
/******/ })
/************************************************************************/
/******/ ([
/* 0 */
/***/ (function(module, exports, __webpack_require__) {

/**
 * Web worker script that uses an emscripten-compiled version
 * of OpenEXR to parse EXR files. The webworker can be used like this:
 *
 *  const ExrParser = require('worker-loader!./utils/exr-parser-webworker.js');
 *  var worker = new ExrParser();
 *  # Send an ArrayBuffer to the webworker
 *  # Because it's basses as the second argument, the data won't be copied
 *  # but ownership will be transfered.
 *  worker.postMessage({ data }, [data]);
 *  worker.onmessage = (event: MessageEvent) => {
 *      if (event.data.success) {
 *          console.log(event.data.image);
 *      } else {
 *          console.error(event.data.message);
 *      }
 *  };
 */


const exrwrapPath = __webpack_require__(1);
const exrwrapWasmPath = __webpack_require__(2);

let openEXRLoaded = false;
let queuedJobs = [];
let OpenEXR;

importScripts(exrwrapPath);

EXR().then(function(Module) {
	OpenEXR = Module;
	openEXRLoaded = true;
	while (queuedJobs.length > 0) {
		const job = queuedJobs.shift();
		if (job) {
			handleJob(job);
		}
	}
});

self.addEventListener('message', (event) => {
    if (!openEXRLoaded) {
        queuedJobs.push(event.data);
    } else {
        handleJob(event.data);
    }
});

function handleJob(job) {
    const jobId = job.jobId;
    try {
        const image = parseExr(job.data);
        self.postMessage(
            {
                jobId,
                success: true,
                image
            },
            [image.data.buffer]
        );
    } catch (error) {
        console.log('Error: ', error);
        self.postMessage({
            jobId,
            success: false,
            message: error.toString()
        });
    }
}

function parseExr(data) {
    console.time('Decoding EXR'); // tslint:disable-line
    let exrImage = null; // tslint:disable-line:no-any
    try {
        exrImage = OpenEXR.loadEXRStr(data);
        const channels = exrImage.channels();
        const {
            width,
            height
        } = exrImage;
        let nChannels = channels.length;
        let exrData;
        if (nChannels === 1) {
            const z = exrImage.plane(exrImage.channels()[0]);
            exrData = new Float32Array(width * height);
            for (let i = 0; i < width * height; i++) {
                exrData[i] = z[i];
            }
        } else if (exrImage.channels().includes('R') &&
            exrImage.channels().includes('G') &&
            exrImage.channels().includes('B')) {
            const r = exrImage.plane('R');
            const g = exrImage.plane('G');
            const b = exrImage.plane('B');
            exrData = new Float32Array(width * height * 3);
            for (let i = 0; i < width * height; i++) {
                exrData[i * 3] = r[i];
                exrData[i * 3 + 1] = g[i];
                exrData[i * 3 + 2] = b[i];
            }
            nChannels = 3;
        } else {
            throw new Error('EXR image not supported');
        }
        return {
            height,
            width,
            nChannels,
            data: exrData,
            type: 'HdrImage',
        };
    } finally {
        if (exrImage) {
            exrImage.delete();
        }
        console.timeEnd('Decoding EXR'); // tslint:disable-line
    }
}


/***/ }),
/* 1 */
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "exr-wrap.js";

/***/ }),
/* 2 */
/***/ (function(module, exports, __webpack_require__) {

module.exports = __webpack_require__.p + "exr-wrap.wasm";

/***/ })
/******/ ]);
//# sourceMappingURL=exr.worker.js.map