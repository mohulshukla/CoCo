"use strict";
/*
 * ATTENTION: An "eval-source-map" devtool has been used.
 * This devtool is neither made for production nor for readable output files.
 * It uses "eval()" calls to create a separate source file with attached SourceMaps in the browser devtools.
 * If you are trying to read the output file, select a different devtool (https://webpack.js.org/configuration/devtool/)
 * or disable the default devtool with "devtool: false".
 * If you are looking for production-ready output files, see mode: "production" (https://webpack.js.org/configuration/mode/).
 */
exports.id = "vendor-chunks/assign-symbols";
exports.ids = ["vendor-chunks/assign-symbols"];
exports.modules = {

/***/ "(ssr)/./node_modules/assign-symbols/index.js":
/*!**********************************************!*\
  !*** ./node_modules/assign-symbols/index.js ***!
  \**********************************************/
/***/ ((module) => {

eval("/*!\n * assign-symbols <https://github.com/jonschlinkert/assign-symbols>\n *\n * Copyright (c) 2015, Jon Schlinkert.\n * Licensed under the MIT License.\n */ \nmodule.exports = function(receiver, objects) {\n    if (receiver === null || typeof receiver === \"undefined\") {\n        throw new TypeError(\"expected first argument to be an object.\");\n    }\n    if (typeof objects === \"undefined\" || typeof Symbol === \"undefined\") {\n        return receiver;\n    }\n    if (typeof Object.getOwnPropertySymbols !== \"function\") {\n        return receiver;\n    }\n    var isEnumerable = Object.prototype.propertyIsEnumerable;\n    var target = Object(receiver);\n    var len = arguments.length, i = 0;\n    while(++i < len){\n        var provider = Object(arguments[i]);\n        var names = Object.getOwnPropertySymbols(provider);\n        for(var j = 0; j < names.length; j++){\n            var key = names[j];\n            if (isEnumerable.call(provider, key)) {\n                target[key] = provider[key];\n            }\n        }\n    }\n    return target;\n};\n//# sourceURL=[module]\n//# sourceMappingURL=data:application/json;charset=utf-8;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoiKHNzcikvLi9ub2RlX21vZHVsZXMvYXNzaWduLXN5bWJvbHMvaW5kZXguanMiLCJtYXBwaW5ncyI6IkFBQUE7Ozs7O0NBS0MsR0FFRDtBQUVBQSxPQUFPQyxPQUFPLEdBQUcsU0FBU0MsUUFBUSxFQUFFQyxPQUFPO0lBQ3pDLElBQUlELGFBQWEsUUFBUSxPQUFPQSxhQUFhLGFBQWE7UUFDeEQsTUFBTSxJQUFJRSxVQUFVO0lBQ3RCO0lBRUEsSUFBSSxPQUFPRCxZQUFZLGVBQWUsT0FBT0UsV0FBVyxhQUFhO1FBQ25FLE9BQU9IO0lBQ1Q7SUFFQSxJQUFJLE9BQU9JLE9BQU9DLHFCQUFxQixLQUFLLFlBQVk7UUFDdEQsT0FBT0w7SUFDVDtJQUVBLElBQUlNLGVBQWVGLE9BQU9HLFNBQVMsQ0FBQ0Msb0JBQW9CO0lBQ3hELElBQUlDLFNBQVNMLE9BQU9KO0lBQ3BCLElBQUlVLE1BQU1DLFVBQVVDLE1BQU0sRUFBRUMsSUFBSTtJQUVoQyxNQUFPLEVBQUVBLElBQUlILElBQUs7UUFDaEIsSUFBSUksV0FBV1YsT0FBT08sU0FBUyxDQUFDRSxFQUFFO1FBQ2xDLElBQUlFLFFBQVFYLE9BQU9DLHFCQUFxQixDQUFDUztRQUV6QyxJQUFLLElBQUlFLElBQUksR0FBR0EsSUFBSUQsTUFBTUgsTUFBTSxFQUFFSSxJQUFLO1lBQ3JDLElBQUlDLE1BQU1GLEtBQUssQ0FBQ0MsRUFBRTtZQUVsQixJQUFJVixhQUFhWSxJQUFJLENBQUNKLFVBQVVHLE1BQU07Z0JBQ3BDUixNQUFNLENBQUNRLElBQUksR0FBR0gsUUFBUSxDQUFDRyxJQUFJO1lBQzdCO1FBQ0Y7SUFDRjtJQUNBLE9BQU9SO0FBQ1QiLCJzb3VyY2VzIjpbIndlYnBhY2s6Ly92aWJlLTNkLWNvZGUvLi9ub2RlX21vZHVsZXMvYXNzaWduLXN5bWJvbHMvaW5kZXguanM/ZWFlYiJdLCJzb3VyY2VzQ29udGVudCI6WyIvKiFcbiAqIGFzc2lnbi1zeW1ib2xzIDxodHRwczovL2dpdGh1Yi5jb20vam9uc2NobGlua2VydC9hc3NpZ24tc3ltYm9scz5cbiAqXG4gKiBDb3B5cmlnaHQgKGMpIDIwMTUsIEpvbiBTY2hsaW5rZXJ0LlxuICogTGljZW5zZWQgdW5kZXIgdGhlIE1JVCBMaWNlbnNlLlxuICovXG5cbid1c2Ugc3RyaWN0JztcblxubW9kdWxlLmV4cG9ydHMgPSBmdW5jdGlvbihyZWNlaXZlciwgb2JqZWN0cykge1xuICBpZiAocmVjZWl2ZXIgPT09IG51bGwgfHwgdHlwZW9mIHJlY2VpdmVyID09PSAndW5kZWZpbmVkJykge1xuICAgIHRocm93IG5ldyBUeXBlRXJyb3IoJ2V4cGVjdGVkIGZpcnN0IGFyZ3VtZW50IHRvIGJlIGFuIG9iamVjdC4nKTtcbiAgfVxuXG4gIGlmICh0eXBlb2Ygb2JqZWN0cyA9PT0gJ3VuZGVmaW5lZCcgfHwgdHlwZW9mIFN5bWJvbCA9PT0gJ3VuZGVmaW5lZCcpIHtcbiAgICByZXR1cm4gcmVjZWl2ZXI7XG4gIH1cblxuICBpZiAodHlwZW9mIE9iamVjdC5nZXRPd25Qcm9wZXJ0eVN5bWJvbHMgIT09ICdmdW5jdGlvbicpIHtcbiAgICByZXR1cm4gcmVjZWl2ZXI7XG4gIH1cblxuICB2YXIgaXNFbnVtZXJhYmxlID0gT2JqZWN0LnByb3RvdHlwZS5wcm9wZXJ0eUlzRW51bWVyYWJsZTtcbiAgdmFyIHRhcmdldCA9IE9iamVjdChyZWNlaXZlcik7XG4gIHZhciBsZW4gPSBhcmd1bWVudHMubGVuZ3RoLCBpID0gMDtcblxuICB3aGlsZSAoKytpIDwgbGVuKSB7XG4gICAgdmFyIHByb3ZpZGVyID0gT2JqZWN0KGFyZ3VtZW50c1tpXSk7XG4gICAgdmFyIG5hbWVzID0gT2JqZWN0LmdldE93blByb3BlcnR5U3ltYm9scyhwcm92aWRlcik7XG5cbiAgICBmb3IgKHZhciBqID0gMDsgaiA8IG5hbWVzLmxlbmd0aDsgaisrKSB7XG4gICAgICB2YXIga2V5ID0gbmFtZXNbal07XG5cbiAgICAgIGlmIChpc0VudW1lcmFibGUuY2FsbChwcm92aWRlciwga2V5KSkge1xuICAgICAgICB0YXJnZXRba2V5XSA9IHByb3ZpZGVyW2tleV07XG4gICAgICB9XG4gICAgfVxuICB9XG4gIHJldHVybiB0YXJnZXQ7XG59O1xuIl0sIm5hbWVzIjpbIm1vZHVsZSIsImV4cG9ydHMiLCJyZWNlaXZlciIsIm9iamVjdHMiLCJUeXBlRXJyb3IiLCJTeW1ib2wiLCJPYmplY3QiLCJnZXRPd25Qcm9wZXJ0eVN5bWJvbHMiLCJpc0VudW1lcmFibGUiLCJwcm90b3R5cGUiLCJwcm9wZXJ0eUlzRW51bWVyYWJsZSIsInRhcmdldCIsImxlbiIsImFyZ3VtZW50cyIsImxlbmd0aCIsImkiLCJwcm92aWRlciIsIm5hbWVzIiwiaiIsImtleSIsImNhbGwiXSwic291cmNlUm9vdCI6IiJ9\n//# sourceURL=webpack-internal:///(ssr)/./node_modules/assign-symbols/index.js\n");

/***/ })

};
;