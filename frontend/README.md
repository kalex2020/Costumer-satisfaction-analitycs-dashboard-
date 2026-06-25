# 📊 Frontend Dashboard Intrak - Documentación Completa

## 🎯 Descripción General

Frontend completamente funcional en **HTML + CSS + JavaScript vanilla** (sin frameworks pesados) que consume la API REST del backend FastAPI para mostrar un dashboard analítico-predictivo de logística de última milla.

**URLs de Acceso:**
- 🌐 Frontend: **http://localhost:3000**
- 🔌 Backend API: **http://localhost:8000**
- 📚 Documentación API: **http://localhost:8000/docs**

---

## 📁 Estructura de Archivos

```
/frontend/
├── index.html          # Estructura HTML principal
├── styles.css          # Estilos CSS con paleta de colores Intrak
├── main.js             # Lógica JavaScript (consumo API, gráficos, filtros)
├── server.py           # Servidor HTTP simple para servir frontend
└── README.md           # Este archivo
```

---

## 🎨 Paleta de Colores Intrak

```css
--color-primary:     #0B3C5D  /* Azul oscuro - Headers, botones activos */
--color-secondary:   #1F6AE1  /* Azul medio - Botones principales */
--color-accent:      #4DA3FF  /* Azul claro - Acentos, detalles */
--color-bg:          #F4F6F9  /* Gris claro - Fondo general */
--color-text:        #2C3E50  /* Gris oscuro - Texto principal */
--color-text-light:  #7F8C8D  /* Gris - Texto secundario */
```

---

## 🏗️ Estructura de Componentes

### 1. **Header (Encabezado)**
- Título: "Intrak, dashboard analítico y predictivo"
- Descripción dinámica según página activa
- Navegación entre 3 vistas:
  - Análisis Actual
  - Análisis Futuro
  - Confianza y Mejora Continua

### 2. **Flashcards (KPIs)**
- Grid de 4 tarjetas principales
- Datos consumidos desde: `GET /metrics/flashcards`
- Indicadores:
  - ETA Promedio
  - ETA Máximo
  - ETA Mínimo
  - N° Entregas

### 3. **Sidebar de Filtros (Izquierda)**
Controles interactivos:
- Selector de Modelo (Linear, GLM, XGBoost)
- Localidad (dropdown)
- Tipo de Vía (Calle, Carrera, Diagonal)
- Vehículo (Moto, Carro, Bicicleta)
- Nivel de Tráfico (slider de 0-5)
- Rango de Fechas
- Botones: Aplicar Filtros / Limpiar

### 4. **Contenido Principal (Grid de Gráficos)**

#### PÁGINA 1: **Análisis Actual**
Visualización del desempeño actual de entregas:
1. **Línea**: Tiempo Real vs Estimado (evolución temporal)
2. **Barras**: Tiempo promedio por localidad
3. **Scatter**: Impacto del tráfico en ETA (con escala de color)
4. **Scatter**: Impacto de la lluvia en ETA
5. **Boxplot**: Distribución de tiempo por tipo de vía

#### PÁGINA 2: **Análisis Futuro**
Predicciones y análisis mediante ML:
1. **Línea**: ETA Real vs ETA Predicho
2. **Histograma**: Distribución del ETA futuro
3. **Barras**: Riesgo por localidad
4. **Heatmap**: Matriz de sensibilidad (tráfico vs lluvia)

#### PÁGINA 3: **Confianza y Mejora Continua**
Métricas de precisión y recomendaciones:
1. **Barras Agrupadas**: MAE, RMSE, MAPE por modelo
2. **Línea**: Evolución del error en el tiempo
3. **Histograma**: Distribución del error de predicción
4. **Cards**: 5 propuestas de mejora documentadas

---

## 🔌 Integración con API

### Endpoints Consumidos

```javascript
// Cargar KPIs
GET http://localhost:8000/metrics/flashcards
→ Retorna: { eta_mean, eta_max, eta_min, n_records }

// Cargar métricas por modelo
GET http://localhost:8000/metrics/models
→ Retorna: { metrics: { linear: {...}, glm: {...}, ... } }

// Hacer predicción
POST http://localhost:8000/predict/eta
Body: {
  localidad: "Kennedy",
  distancia_km: 5.5,
  nivel_trafico: 3,
  lluvia_mm: 2.0,
  estrato: 3,
  tipo_via: "Calle",
  tipo_zona: "Urbana",
  vehiculo: "Moto",
  tiempo_estimado_min: 15.0
}
→ Retorna: { prediction, model_name, confidence }
```

### Manejo de Errores

Todos los fetch tienen try/catch:
```javascript
try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    // Procesar datos
} catch (error) {
    console.error('Error:', error);
    showError('Mensaje al usuario', container);
}
```

---

## 📊 Librerías de Gráficos

### Plotly.js (Primaria)
Utilizada para gráficos complejos:
- Líneas con múltiples series
- Scatter con escala de colores
- Boxplots
- Heatmaps
- Histogramas

```html
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
```

**Ventajas:**
- Interactividad nativa (zoom, pan, hover)
- Gráficos responsivos
- Sin configuración de canvas

### Chart.js (Alternativa)
Disponible para gráficos simples si se requiere.

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.js"></script>
```

---

## 🛠️ Stack Tecnológico

| Componente | Tecnología | Razón |
|-----------|-----------|--------|
| Estructura | HTML 5 | Semántico, accesible |
| Estilos | CSS 3 | Grid, Flexbox, variables CSS |
| Lógica | JavaScript ES6+ | Fetch API, async/await, módulos |
| Gráficos | Plotly.js | Interactividad, responsividad |
| Servidor | Python HTTP | Simple, sin dependencias |
| Estado | Object JSON | Gestión centralizada |

---

## 🚀 Características Implementadas

✅ **Navegación por páginas** - Sin recargas (SPA simple)
✅ **Filtros interactivos** - Actualización dinámica de datos
✅ **Gráficos responsivos** - Adaptables a cualquier resolución
✅ **Consumo de API** - Fetch con manejo de errores
✅ **Paleta corporativa** - Azules Intrak consistentes
✅ **Código comentado** - Documentación inline
✅ **Estructura modular** - Funciones independientes
✅ **Mobile responsive** - Media queries hasta 480px
✅ **Loading states** - Skeletons y spinners
✅ **Error handling** - Mensajes claros al usuario

---

## 💡 Funcionalidades Principales

### 1. Cargar datos en tiempo real
```javascript
await loadFlashcards();  // Actualiza KPIs
await loadPageGraphics(); // Recarga gráficos de página actual
```

### 2. Filtrar por múltiples campos
```javascript
const filters = {
    localidad: 'Kennedy',
    via: 'Calle',
    vehiculo: 'Moto',
    traffic: 3
};
```

### 3. Navegar entre vistas
```javascript
// Click en botón → Cambia página + recarga gráficos
navButton.addEventListener('click', async (e) => {
    appState.currentPage = e.target.dataset.page;
    await loadPageGraphics();
});
```

### 4. Renderizar gráficos dinámicos
```javascript
Plotly.newPlot('chart-container', data, layout, {
    responsive: true,
    displayModeBar: false
});
```

---

## 🎯 Buenas Prácticas Implementadas

### Estructura de Código
- ✅ Separación clara HTML/CSS/JS
- ✅ Funciones pequeñas y específicas
- ✅ Nombres descriptivos (camelCase)
- ✅ Comentarios explicativos

### Performance
- ✅ Async/await para operaciones I/O
- ✅ Carga de librerías desde CDN
- ✅ Caché implícita de datos
- ✅ Transiciones CSS fluidas

### Seguridad
- ✅ Validación de respuestas HTTP
- ✅ Manejo de errores CORS
- ✅ No hardcodear variables sensibles
- ✅ Sanitización implícita con textContent

### UX/UI
- ✅ Paleta consistente
- ✅ Feedback visual en interacciones
- ✅ Estados de carga visible
- ✅ Mensajes de error claros

---

## 🔧 Configuración y Despliegue

### Desarrollo Local

```bash
# Terminal 1: Backend
cd K:\Dashboard Delivery\backend
conda activate dashboard
python main.py  # Puerto 8000

# Terminal 2: Frontend
cd K:\Dashboard Delivery\frontend
conda activate dashboard
python server.py  # Puerto 3000

# Abrir navegador
http://localhost:3000
```

### Variables de Configuración

Editar en `main.js`:
```javascript
const API_BASE_URL = 'http://localhost:8000';  // URL del backend
const REFRESH_INTERVAL = 30000;  // Milisegundos
```

### Cambiar Puertos

**Frontend** (server.py):
```python
PORT = 3000  # Cambiar aquí
```

**Backend** (main.py):
```python
# En main.py del backend
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Puerto aquí
```

---

## 📱 Responsividad

### Breakpoints
- **Desktop**: > 1400px (3+ columnas)
- **Tablet**: 768px - 1400px (2 columnas)
- **Mobile**: < 768px (1 columna)
- **Teléfono**: < 480px (optimizado)

### Elementos Adaptables
- ✅ Grid de gráficos → columnas dinámicas
- ✅ Sidebar → Ancho fijo o full-width
- ✅ Flashcards → 4 → 2 → 1 columna
- ✅ Tipografía → Reduce en móvil

---

## 🐛 Troubleshooting

### Problema: CORS Error
**Solución**: Asegúrate que el backend tiene CORS habilitado
```python
# En backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Problema: API no responde
**Solución**: Verifica que backend está corriendo
```bash
# En terminal
curl http://localhost:8000/health
# Debe retornar JSON con status "ok"
```

### Problema: Gráficos no cargan
**Solución**: Abre Console (F12) y busca errores
```javascript
// Verifica en Console:
> Plotly
// Debe retornar objeto, no undefined
```

### Problema: Filtros no actualizan
**Solución**: Verifica que API retorna datos con filtros
```javascript
// En Console:
await applyFilters();
// Debe actualizar estado y UI
```

---

## 📈 Próximas Mejoras Opcionales

- [ ] Exportar gráficos a PDF
- [ ] Temas oscuro/claro
- [ ] Notificaciones push de anomalías
- [ ] Caché con IndexedDB
- [ ] PWA (Progressive Web App)
- [ ] WebSockets para datos en tiempo real
- [ ] Autenticación y roles de usuario
- [ ] Dashboard customizable (drag-drop widgets)

---

## 📞 Soporte

**Errores encontrados:**
1. Revisar Console (F12 → Console)
2. Verificar conectividad: `curl http://localhost:8000/health`
3. Revisar logs de backend en terminal

**Contacto:**
- Frontend: `main.js` - Lógica
- Estilos: `styles.css` - Diseño
- API: Ver documentación en `http://localhost:8000/docs`

---

## ✅ Checklist de Funcionalidad

- [x] Navegación entre 3 páginas
- [x] Flashcards con datos de API
- [x] Filtros interactivos (6 tipos)
- [x] 13 gráficos diferentes
- [x] Responsive design
- [x] Paleta de colores Intrak
- [x] Manejo de errores
- [x] Código comentado
- [x] Separación HTML/CSS/JS
- [x] Sin frameworks pesados

---

**Generado:** 12/04/2026
**Version:** 1.0
**Status:** ✅ PRODUCCIÓN
