const SCALE = 2;

function runWithFPS(fn, fps) {
  let interval = 1000 / fps;
  let then = Date.now();
  let stopped = false;
  let animationFrame;

  function run() {
    if (!stopped) {
      animationFrame = requestAnimationFrame(run);
    }

    var now = Date.now();
    var delta = now - then;

    if (delta > interval && !stopped) {
      then = now - (delta % interval);
      fn(delta);
    }
  }

  animationFrame = requestAnimationFrame(run);
  fn(0);

  return function () {
    stopped = true;
    cancelAnimationFrame(animationFrame);
  };
}

const downloadLink = document.createElement('a');
document.body.appendChild(downloadLink);
downloadLink.style = 'display: none';
function downloadCanvas(canvas, render, fn) {
  return new Promise((resolve) => {
    render();
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      downloadLink.href = url;
      downloadLink.download = `${fn}.png`;
      downloadLink.click();
      URL.revokeObjectURL(url);
      resolve();
    }, 'image/png');
  });
}

function createShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('Error compiling shader:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function createProgram(gl, vertexShader, fragmentShader) {
  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Error linking program:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }
  return program;
}

function initAIBackground(root, options) {
  const canvas = document.createElement('canvas');
  root.appendChild(canvas);
  canvas.style.width = '100%';
  canvas.style.height = '100%';
  canvas.style.display = 'block';

  const gl = canvas.getContext('webgl');
  if (!gl) {
    console.error('WebGL not supported');
    return;
  }

  const vertexShaderSource = `
      attribute vec2 aPosition;
      attribute vec2 aTexCoord;
      uniform vec2 uSize;
      uniform float uLineWidth;
      varying vec2 vUv;
      varying float uDensity;
      varying float uRimWidth;

      void main() {
        uDensity = 1.0 / (uSize.x / uLineWidth);
        uRimWidth = (uLineWidth / 35.0) / uSize.x;

        vUv = aTexCoord;
        vUv.y = 1.0 - vUv.y;
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }
  `;

  const fragmentShaderSource = `
      #define PI 3.1415926535897932384626433832795

      precision highp float;
      varying vec2 vUv;
      varying float uDensity;
      varying float uRimWidth;
      uniform sampler2D uTexture;
      uniform sampler2D uDispMap;
      uniform float uTime;
      uniform float uOffset;
      uniform float uAspectRatio;
      uniform float uCurvature;
      uniform float uDistortion;
      uniform vec2 uSize;

      //
      // psrdnoise2.glsl
      //
      // Authors: Stefan Gustavson (stefan.gustavson@gmail.com)
      // and Ian McEwan (ijm567@gmail.com)
      // Version 2021-12-02, published under the MIT license (see below)
      //
      // Copyright (c) 2021 Stefan Gustavson and Ian McEwan.
      //
      // Permission is hereby granted, free of charge, to any person obtaining a
      // copy of this software and associated documentation files (the "Software"),
      // to deal in the Software without restriction, including without limitation
      // the rights to use, copy, modify, merge, publish, distribute, sublicense,
      // and/or sell copies of the Software, and to permit persons to whom the
      // Software is furnished to do so, subject to the following conditions:
      //
      // The above copyright notice and this permission notice shall be included
      // in all copies or substantial portions of the Software.
      //
      // THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
      // IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
      // FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
      // THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
      // LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
      // FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
      // DEALINGS IN THE SOFTWARE.
      //
      
      //
      // Periodic (tiling) 2-D simplex noise (hexagonal lattice gradient noise)
      // with rotating gradients and analytic derivatives.
      //
      // This is (yet) another variation on simplex noise. Unlike previous
      // implementations, the grid is axis-aligned and slightly stretched in
      // the y direction to permit rectangular tiling.
      // The noise pattern can be made to tile seamlessly to any integer period
      // in x and any even integer period in y. Odd periods may be specified
      // for y, but then the actual tiling period will be twice that number.
      //
      // The rotating gradients give the appearance of a swirling motion, and
      // can serve a similar purpose for animation as motion along z in 3-D
      // noise. The rotating gradients in conjunction with the analytic
      // derivatives allow for "flow noise" effects as presented by Ken
      // Perlin and Fabrice Neyret.
      //
      
      
      //
      // 2-D tiling simplex noise with rotating gradients and analytical derivative.
      // "vec2 x" is the point (x,y) to evaluate,
      // "vec2 period" is the desired periods along x and y, and
      // "float alpha" is the rotation (in radians) for the swirling gradients.
      // The "float" return value is the noise value, and
      // the "out vec2 gradient" argument returns the x,y partial derivatives.
      //
      // Setting either period to 0.0 or a negative value will skip the wrapping
      // along that dimension. Setting both periods to 0.0 makes the function
      // execute about 15% faster.
      //
      // Not using the return value for the gradient will make the compiler
      // eliminate the code for computing it. This speeds up the function
      // by 10-15%.
      //
      // The rotation by alpha uses one single addition. Unlike the 3-D version
      // of psrdnoise(), setting alpha == 0.0 gives no speedup.
      //
      float psrdnoise(vec2 x, vec2 period, float alpha, out vec2 gradient) {
        // Transform to simplex space (axis-aligned hexagonal grid)
        vec2 uv = vec2(x.x + x.y*0.5, x.y);
    
        // Determine which simplex we're in, with i0 being the "base"
        vec2 i0 = floor(uv);
        vec2 f0 = fract(uv);
        // o1 is the offset in simplex space to the second corner
        float cmp = step(f0.y, f0.x);
        vec2 o1 = vec2(cmp, 1.0-cmp);
    
        // Enumerate the remaining simplex corners
        vec2 i1 = i0 + o1;
        vec2 i2 = i0 + vec2(1.0, 1.0);
    
        // Transform corners back to texture space
        vec2 v0 = vec2(i0.x - i0.y * 0.5, i0.y);
        vec2 v1 = vec2(v0.x + o1.x - o1.y * 0.5, v0.y + o1.y);
        vec2 v2 = vec2(v0.x + 0.5, v0.y + 1.0);
    
        // Compute vectors from v to each of the simplex corners
        vec2 x0 = x - v0;
        vec2 x1 = x - v1;
        vec2 x2 = x - v2;
    
        vec3 iu, iv;
        vec3 xw, yw;
    
        // Wrap to periods, if desired
        if(any(greaterThan(period, vec2(0.0)))) {
            xw = vec3(v0.x, v1.x, v2.x);
            yw = vec3(v0.y, v1.y, v2.y);
            if(period.x > 0.0)
                xw = mod(vec3(v0.x, v1.x, v2.x), period.x);
            if(period.y > 0.0)
                yw = mod(vec3(v0.y, v1.y, v2.y), period.y);
            // Transform back to simplex space and fix rounding errors
            iu = floor(xw + 0.5*yw + 0.5);
            iv = floor(yw + 0.5);
        } else { // Shortcut if neither x nor y periods are specified
            iu = vec3(i0.x, i1.x, i2.x);
            iv = vec3(i0.y, i1.y, i2.y);
        }
    
        // Compute one pseudo-random hash value for each corner
        vec3 hash = mod(iu, 289.0);
        hash = mod((hash*51.0 + 2.0)*hash + iv, 289.0);
        hash = mod((hash*34.0 + 10.0)*hash, 289.0);
    
        // Pick a pseudo-random angle and add the desired rotation
        vec3 psi = hash * 0.07482 + alpha;
        vec3 gx = cos(psi);
        vec3 gy = sin(psi);
    
        // Reorganize for dot products below
        vec2 g0 = vec2(gx.x,gy.x);
        vec2 g1 = vec2(gx.y,gy.y);
        vec2 g2 = vec2(gx.z,gy.z);
    
        // Radial decay with distance from each simplex corner
        vec3 w = 0.8 - vec3(dot(x0, x0), dot(x1, x1), dot(x2, x2));
        w = max(w, 0.0);
        vec3 w2 = w * w;
        vec3 w4 = w2 * w2;
    
        // The value of the linear ramp from each of the corners
        vec3 gdotx = vec3(dot(g0, x0), dot(g1, x1), dot(g2, x2));
    
        // Multiply by the radial decay and sum up the noise value
        float n = dot(w4, gdotx);
    
        // Compute the first order partial derivatives
        vec3 w3 = w2 * w;
        vec3 dw = -8.0 * w3 * gdotx;
        vec2 dn0 = w4.x * g0 + dw.x * x0;
        vec2 dn1 = w4.y * g1 + dw.y * x1;
        vec2 dn2 = w4.z * g2 + dw.z * x2;
        gradient = 10.9 * (dn0 + dn1 + dn2);
    
        // Scale the return value to fit nicely into the range [-1,1]
        return 10.9 * n;
      }

      float quinticIn(float t) { return pow(t, 5.0); }

      float quadIn(float t) { return pow(t, 1.2); }

      float bounceOut(in float t) {
        const float a = 4.0 / 11.0;
        const float b = 8.0 / 11.0;
        const float c = 9.0 / 10.0;
    
        const float ca = 4356.0 / 361.0;
        const float cb = 35442.0 / 1805.0;
        const float cc = 16061.0 / 1805.0;
    
        float t2 = t * t;
    
        return t < a
            ? 7.5625 * t2
            : t < b
                ? 9.075 * t2 - 9.9 * t + 3.4
                : t < c
                    ? ca * t2 - cb * t + cc
                    : 10.8 * t * t - 20.52 * t + 10.72;
      }

      float bounceInOut(in float t) {
        return t < 0.5
            ? 0.5 * (1.0 - bounceOut(1.0 - t * 2.0))
            : 0.5 * bounceOut(t * 2.0 - 1.0) + 0.5;
      }

      float bounceIn(in float t) { return 1.0 - bounceOut(1.0 - t); }

      float cubicIn(in float t) { return t * t * t; }

      float curve(float t, float d) {
        if (smoothstep(1.0, d, t) == 1.0) {
          return cubicIn(1.0 - t / d) * 1.3;
        }
        return quadIn(t);
      }

      float glare(float t) {
        return quinticIn(t);
      }

      float wave (float t) { return abs(0.5 - t) / 0.5; }

      float sigmoid(float x) {
        if (x >= 1.0) return 1.0;
        else if (x <= -1.0) return 0.0;
        else return 0.5 + x * (1.0 - abs(x) * 0.5);
      }

      void main() {
        vec2 st = vUv;
        vec2 gradient;
        vec2 gradient2;
        float n = psrdnoise(vec2(3.) * st, vec2(0.), 12.6 * uTime, gradient);
        float n2 = psrdnoise(10000. * st, vec2(1000.), 32.6 * uTime, gradient2);
        float t = uTime * 0.05;
        float curveRate = PI * 0.24 + t * 0.02 + 1.35;
        float rate = ((1.0 - st.y) * 1. * uCurvature + curveRate) * 2.0;
        float linesRate = uDensity * 0.2 + sin(uTime) * 0.009;
        float base = (st.x - 0.5) + cubicIn(sigmoid(cos(rate) * 0.6));
        float lineIndex = floor(base / linesRate);
        // float shade_of_grey = mod(base, linesRate) / linesRate;
        float shade_of_grey = smoothstep(lineIndex * linesRate, (lineIndex + 1.00) * linesRate, base);
        float curveGray = curve(1.0 - shade_of_grey, uRimWidth);
        float glareGray = glare(shade_of_grey);
        vec4 dispColor = vec4(vec3(curveGray), 1.0);
        vec4 glareColor = vec4(vec3((smoothstep(0.9, 0.1, curve(1.0 - shade_of_grey, 0.0)))), 1.0);
        
        vec4 disp = (1.0 - dispColor) * uOffset;
        vec4 color = texture2D(uTexture, st + (n * uDistortion) + vec2(disp.x - 0.1, disp.y - 0.1) * 0.3 + n2 * 0.02 * disp.y);

        gl_FragColor = color;
        gl_FragColor -= glareColor * 0.06;
        gl_FragColor += vec4(glareGray * 0.03 + n2 * 0.02);

        if (shade_of_grey < uRimWidth * 0.07) {
          gl_FragColor *= vec4(vec3(1.0 - 0.04 * sin(rate)), 1.0);
        }

        gl_FragColor = vec4(
          clamp(gl_FragColor.r, 0.0, 1.0),
          clamp(gl_FragColor.g, 0.0, 1.0),
          clamp(gl_FragColor.b, 0.0, 1.0),
          1.0
        );
      }
  `;

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(
    gl,
    gl.FRAGMENT_SHADER,
    fragmentShaderSource
  );

  const program = createProgram(gl, vertexShader, fragmentShader);
  gl.useProgram(program);
  const uTimeLocation = gl.getUniformLocation(program, 'uTime');
  const uAspectRatioLocation = gl.getUniformLocation(program, 'uAspectRatio');
  const uOffsetLocation = gl.getUniformLocation(program, 'uOffset');
  const uCurvatureLocation = gl.getUniformLocation(program, 'uCurvature');
  const uDistortionLocation = gl.getUniformLocation(program, 'uDistortion');
  const uSizeLocation = gl.getUniformLocation(program, 'uSize');
  const uLineWidthLocation = gl.getUniformLocation(program, 'uLineWidth');

  function setCanvasSizeMultiplier(n) {
    const { width, height } = canvas.getBoundingClientRect();
    canvas.width = width * n;
    canvas.height = height * n;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.uniform1f(uAspectRatioLocation, width / height);

    gl.uniform2f(uSizeLocation, width * n, height * n);
  }

  function handleResize() {
    setCanvasSizeMultiplier(SCALE);
  }

  handleResize();
  window.addEventListener('resize', handleResize);

  // плоскость с текстурными координатами
  const positions = new Float32Array([
    -1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, 1.0, 1.0,
    1.0, 1.0,
  ]);

  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

  const aPosition = gl.getAttribLocation(program, 'aPosition');
  gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 4 * 4, 0);
  gl.enableVertexAttribArray(aPosition);

  const aTexCoord = gl.getAttribLocation(program, 'aTexCoord');
  gl.vertexAttribPointer(aTexCoord, 2, gl.FLOAT, false, 4 * 4, 2 * 4);
  gl.enableVertexAttribArray(aTexCoord);

  const indices = new Uint16Array([0, 1, 2, 2, 1, 3]);

  const indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);

  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);

  function setTexture(url) {
    const image = new Image();
    image.src = url;
    image.onload = function () {
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        image
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

      ready = true;
    };
  }

  const registeredOptions = {};
  function initOption(id, { location, cb, transformer = (a) => a }) {
    const updateValue = (value) => {
      if (location) {
        gl.uniform1f(location, transformer(value));
      }
      cb?.(transformer(value));
    };
    registeredOptions[id] = updateValue;
    updateValue(options[id]);
  }

  initOption('refraction', {
    location: uDistortionLocation,
    transformer: Number,
  });
  initOption('offset', { location: uOffsetLocation, transformer: Number });
  initOption('width', {
    location: uLineWidthLocation,
    transformer(n) {
      return Number(n) * 10 * SCALE;
    },
  });
  initOption('curvature', {
    location: uCurvatureLocation,
    transformer: Number,
  });
  let speedRate = 0;
  initOption('speed', {
    cb: (value) => {
      speedRate = value;
    },
    transformer: Number,
  });
  initOption('background', {
    cb: (value) => {
      setTexture(value);
    },
  });
  let fps = 30;
  initOption('fps', {
    transformer: Number,
    cb: (value) => {
      if (!isNaN(value)) {
        fps = value;
      }
    },
  });

  let time = -15;
  let ready = false;
  function render() {
    if (!ready) {
      return;
    }
    time += 0.005 * speedRate * (120 / fps);
    // console.log(time);
    gl.uniform1f(uTimeLocation, time);

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
  }

  const stop = runWithFPS(render, fps);
  return {
    stop,
    canvas,
    render,
    updateOption(id, value) {
      registeredOptions[id]?.(value);
    },
  };
}

const bg = document.querySelector('.bg');

function initPanel(id) {
  const root = document.getElementById(id);
  const input = root.querySelector('input');
  const label = root.querySelector('span');

  const LS_KEY = `value_${id}`;
  if (localStorage.getItem(LS_KEY)) {
    input.value = localStorage.getItem(LS_KEY);
  }
  function update() {
    label.innerText = input.value;
    localStorage.setItem(LS_KEY, input.value);
    aiBackground.updateOption(id, input.value);
  }

  input.addEventListener('input', (e) => {
    update();
  });

  update();
}

const aiBackground = initAIBackground(bg, {
  refraction: 0.16,
  offset: 0.18,
  speed: 0.03,
  curvature: 1.0,
  width: 55,
  fps: 30,
  background: 'texture2.jpg',
});

initPanel('refraction');
initPanel('offset');
initPanel('speed');
initPanel('curvature');
initPanel('width');

aiBackground.canvas.addEventListener('dragover', (e) => {
  e.preventDefault();
});
aiBackground.canvas.addEventListener('drop', (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length) {
    aiBackground.updateOption(
      'background',
      URL.createObjectURL(e.dataTransfer.files[0])
    );
  }
});

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

document.getElementById('download').addEventListener('click', async (e) => {
  e.preventDefault();
  await delay(50);
  await downloadCanvas(
    aiBackground.canvas,
    aiBackground.render,
    'ripley-' + Date.now()
  );
});
