/* ========================================================================
   DASHBOARD INTRAK - LÓGICA PRINCIPAL
   - Consumo de API REST
   - Gestión de navegación por páginas
   - Actualización de flashcards
   - Generación de gráficos
   - Manejo de filtros
   ======================================================================== */

// ========================================================================
// CONFIGURACIÓN GLOBAL
// ========================================================================

const API_BASE_URL = 'http://localhost:8000';
const DASHBOARD_CONFIG = {
    refreshInterval: 30000, // 30 segundos
    defaultModel: 'linear',
    pageDescriptions: {
        actual: 'Análisis del desempeño actual de entregas de última milla en Bogotá',
        futuro: 'Predicciones y análisis futuro del desempeño de entregas',
        confianza: 'Métricas de confianza de los modelos y propuestas de mejora continua'
    }
};

// Estado global
let appState = {
    currentPage: 'actual',
    currentModel: '',
    filters: {
        localidad: '',
        via: '',
        vehiculo: '',
        traffic: 2,
        dateFrom: null,
        dateTo: null
    },
    data: {
        flashcards: null,
        metrics: null,
        predictions: null
    }
};

// ========================================================================
// INICIALIZACIÓN
// ========================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Inicializando Dashboard Intrak...');
    
    // Inicializar navegación
    initializeNavigation();
    
    // Cargar datos iniciales
    await loadFlashcards();
    
    // Inicializar eventos de filtros
    initializeFilters();
    
    // Cargar gráficos de la página actual
    await loadPageGraphics();
    
    setTimeout(() => {
        resizeVisibleCharts();
    }, 500);

    window.addEventListener('resize', resizeVisibleCharts);
    
    console.log('✅ Dashboard inicializado correctamente');
});

// ========================================================================
// NAVEGACIÓN ENTRE PÁGINAS
// ========================================================================

function initializeNavigation() {
    const navButtons = document.querySelectorAll('.nav-button');
    
    navButtons.forEach(button => {
        button.addEventListener('click', async (e) => {
            const page = e.target.dataset.page;
            
            if (appState.currentPage === page) return;
            
            // Actualizar estado
            appState.currentPage = page;
            
            // Actualizar botones
            navButtons.forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');
            
            // Actualizar descripción
            updatePageDescription(page);
            
            // Actualizar secciones visibles
            document.querySelectorAll('.dashboard-page').forEach(section => {
                section.classList.remove('active');
            });
            document.getElementById(`page-${page}`).classList.add('active');
            
            // Cargar KPI y gráficos
            await loadFlashcards();
            await loadPageGraphics();
            
            // Forzar resize de gráficos Plotly al cambiar de pestaña
            setTimeout(() => {
                resizeVisibleCharts();
            }, 100);
        });
    });
}

function updatePageDescription(page) {
    const description = document.getElementById('pageDescription');
    description.textContent = DASHBOARD_CONFIG.pageDescriptions[page] || '';
}

function getNextMonthHorizon() {
    const now = new Date();
    const year = now.getMonth() === 11 ? now.getFullYear() + 1 : now.getFullYear();
    const month = now.getMonth() === 11 ? 1 : now.getMonth() + 2;
    return `${year}-${String(month).padStart(2, '0')}`;
}

function renderPlot(id, data, layout) {
    const container = document.getElementById(id);
    if (!container || !window.Plotly) return;

    const width = Math.max(320, container.clientWidth || 320);
    const height = Math.max(280, container.clientHeight || 280);

    const safeLayout = {
        autosize: true,
        margin: { l: 48, r: 24, t: 20, b: 52 },
        ...layout,
        width,
        height
    };

    Plotly.react(id, data, safeLayout, { responsive: true, displayModeBar: false });

    requestAnimationFrame(() => {
        const node = document.getElementById(id);
        if (node) {
            Plotly.Plots.resize(node);
        }
    });
}

function resizeVisibleCharts() {
    if (!window.Plotly) return;

    document.querySelectorAll('.dashboard-page.active .chart-box').forEach((box) => {
        if (box.id) {
            Plotly.Plots.resize(box);
        }
    });
}

// ========================================================================
// CARGA DE FLASHCARDS (KPIs)
// ========================================================================

async function loadFlashcards() {
    try {
        const flashcardsContainer = document.getElementById('flashcardsContainer');
        
        // Mostrar skeleton
        showLoading(flashcardsContainer);
        
        // Seleccionar endpoint basado en la página actual
        let endpoint = '/metrics/kpis/actual';
        if (appState.currentPage === 'futuro') {
            endpoint = '/metrics/kpis/future';
        } else if (appState.currentPage === 'confianza') {
            endpoint = '/metrics/kpis/confidence';
        }
        
        // Llamar API simplificada
        const model = appState.filters.model || 'linear';
        const month = getNextMonthHorizon();
        const query = appState.currentPage === 'futuro'
            ? `?modelo=${model}&month=${month}`
            : appState.currentPage === 'confianza'
                ? ''
                : `?modelo=${model}`;
        const response = await fetch(`${API_BASE_URL}${endpoint}${query}`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        appState.data.flashcards = data;
        
        // Renderizar flashcards
        renderFlashcards(data);
        
    } catch (error) {
        console.error('❌ Error cargando flashcards:', error);
        showError('No se pudieron cargar los KPIs', document.getElementById('flashcardsContainer'));
    }
}

function renderFlashcards(data) {
    const container = document.getElementById('flashcardsContainer');
    container.innerHTML = '';

    let flashcardsData;
    if (appState.currentPage === 'futuro') {
        flashcardsData = [
            {
                label: 'Pedidos Estimados (Mes)',
                value: data.demand_month_predicted?.toFixed ? data.demand_month_predicted.toFixed(0) : (data.demand_month_predicted ?? 'N/A'),
                unit: 'pedidos',
                icon: '📦'
            },
            {
                label: 'ETA Promedio Esperado',
                value: data.eta_mean?.toFixed(1) ?? 'N/A',
                unit: 'minutos',
                icon: '⏱️'
            },
            {
                label: 'OTIF Esperado',
                value: data.otif?.toFixed(1) ?? 'N/A',
                unit: '%',
                icon: '✅'
            },
            {
                label: '% Retrasos Esperados',
                value: data.delays_pct?.toFixed(1) ?? 'N/A',
                unit: '%',
                icon: '⚠️'
            },
            {
                label: 'Horizonte',
                value: data.horizon ?? getNextMonthHorizon(),
                unit: 'mes',
                icon: '🗓️'
            }
        ];
    } else if (appState.currentPage === 'confianza') {
        flashcardsData = [
            {
                label: 'Precisión Global (4 modelos)',
                value: data.precision_global?.toFixed(2) ?? 'N/A',
                unit: '%',
                icon: '🎯'
            },
            {
                label: 'Error MAE Global',
                value: data.mae_global?.toFixed(3) ?? 'N/A',
                unit: 'min',
                icon: '📉'
            },
            {
                label: 'Error RMSE Global',
                value: data.rmse_global?.toFixed(3) ?? 'N/A',
                unit: 'min',
                icon: '📊'
            },
            {
                label: 'Error MAPE Global',
                value: data.mape_global?.toFixed(2) ?? 'N/A',
                unit: '%',
                icon: '⚠️'
            },
            {
                label: 'Score de Confianza',
                value: data.score_confianza?.toFixed(2) ?? 'N/A',
                unit: '/100',
                icon: '🛡️'
            },
            {
                label: 'Mejor Modelo',
                value: (data.best_model || 'N/A').toUpperCase(),
                unit: `RMSE ${data.best_rmse?.toFixed ? data.best_rmse.toFixed(3) : 'N/A'}`,
                icon: '🏆'
            }
        ];
    } else {
        flashcardsData = [
            {
                label: 'ETA Promedio',
                value: data.eta_mean?.toFixed(1) ?? 'N/A',
                unit: 'minutos',
                icon: '⏱️'
            },
            {
                label: 'OTIF (On-Time)',
                value: data.otif?.toFixed(1) ?? 'N/A',
                unit: '%',
                icon: '✅'
            },
            {
                label: '% Retrasos',
                value: data.delays_pct?.toFixed(1) ?? 'N/A',
                unit: '%',
                icon: '⚠️'
            },
            {
                label: 'N° Entregas',
                value: data.n_records ?? 'N/A',
                unit: 'registros',
                icon: '📦'
            }
        ];
    }
    
    flashcardsData.forEach(card => {
        const flashcard = document.createElement('div');
        flashcard.className = 'flashcard';
        flashcard.innerHTML = `
            <div class="flashcard-label">${card.icon} ${card.label}</div>
            <div class="flashcard-value">${card.value}</div>
            <div class="flashcard-unit">${card.unit}</div>
        `;
        container.appendChild(flashcard);
    });
}

// ========================================================================
// ACTUALIZACIÓN DE GRÁFICOS POR PÁGINA
// ========================================================================

async function loadPageGraphics() {
    try {
        switch (appState.currentPage) {
            case 'actual':
                await loadAnalisisActual();
                break;
            case 'futuro':
                await loadAnalisisFuturo();
                break;
            case 'confianza':
                await loadConfianzaMejora();
                break;
        }
    } catch (error) {
        console.error('❌ Error cargando gráficos:', error);
    }
}

// ========================================================================
// PÁGINA 1: ANÁLISIS ACTUAL
// ========================================================================

async function loadAnalisisActual() {
    console.log('📊 Cargando Análisis Actual...');
    
    try {
        // Obtener datos del backend
        const modelo = appState.filters.model || 'linear';
        const response = await fetch(`${API_BASE_URL}/analytics/actual?modelo=${modelo}`);
        const result = await response.json();
        const data = result.data || {};
        
        // Gráfico 1: Tiempo Real vs Estimado
        if (data.evolucion_temporal) {
            createTimeVsEstimateChart(data.evolucion_temporal);
        } else {
            createTimeVsEstimateChart();
        }
        
        // Gráfico 2: Tiempo por Localidad
        if (data.por_localidad) {
            createTimeByLocalidadChart(data.por_localidad);
        } else {
            createTimeByLocalidadChart();
        }
        
        // Gráfico 3: Impacto del Tráfico
        if (data.trafico_impacto) {
            createTrafficImpactChart(data.trafico_impacto);
        } else {
            createTrafficImpactChart();
        }
        
        // Gráfico 4: Impacto de la Lluvia
        if (data.lluvia_impacto) {
            createRainImpactChart(data.lluvia_impacto);
        } else {
            createRainImpactChart();
        }
        
        // Gráfico 5: Tiempo por Tipo de Vía
        if (data.por_via) {
            createTimeByViaChart(data.por_via);
        } else {
            createTimeByViaChart();
        }
        
    } catch (error) {
        console.error('Error en Análisis Actual:', error);
        // Cargar con datos por defecto
        createTimeVsEstimateChart();
        createTimeByLocalidadChart();
        createTrafficImpactChart();
        createRainImpactChart();
        createTimeByViaChart();
    }
}

function createTimeVsEstimateChart(data = null) {
    let plotData;
    
    if (data && data.real && data.estimado) {
        plotData = [
            {
                x: data.x || Array.from({length: data.real.length}, (_, i) => i+1),
                y: data.real,
                name: 'Tiempo Real',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#1F6AE1', width: 3 },
                marker: { size: 8, color: '#1F6AE1' }
            },
            {
                x: data.x || Array.from({length: data.estimado.length}, (_, i) => i+1),
                y: data.estimado,
                name: 'Tiempo Estimado',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#4DA3FF', width: 3, dash: 'dash' },
                marker: { size: 8, color: '#4DA3FF' }
            }
        ];
    } else {
        plotData = [
            {
                x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                y: [45, 52, 48, 61, 55, 58, 62, 59, 63, 65],
                name: 'Tiempo Real',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#1F6AE1', width: 3 },
                marker: { size: 8, color: '#1F6AE1' }
            },
            {
                x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                y: [42, 48, 46, 55, 50, 52, 58, 54, 58, 62],
                name: 'Tiempo Estimado',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#4DA3FF', width: 3, dash: 'dash' },
                marker: { size: 8, color: '#4DA3FF' }
            }
        ];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Entregas (n)' },
        yaxis: { title: 'Tiempo (minutos)' },
        hovermode: 'x unified',
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }
    };
    
    renderPlot('chart-timeVsEstimate', plotData, layout);
}

function createTimeByLocalidadChart(data = null) {
    let plotData;
    
    if (data && Array.isArray(data) && data.length > 0) {
        plotData = [{
            x: data.map(d => d.localidad),
            y: data.map(d => d['mean'] || d.mean || 0),
            type: 'bar',
            marker: { color: '#0B3C5D' }
        }];
    } else {
        plotData = [{
            x: ['Kennedy', 'Ciudad Bolívar', 'Soacha', 'Usaquén', 'Chapinero'],
            y: [62.3, 58.1, 71.5, 48.2, 52.8],
            type: 'bar',
            marker: { color: '#0B3C5D' }
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Localidad' },
        yaxis: { title: 'Tiempo Promedio (min)' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    renderPlot('chart-timeByLocalidad', plotData, layout);
}

function createTrafficImpactChart(data = null) {
    let plotData;
    
    if (data && Array.isArray(data) && data.length > 0) {
        const allPoints = [];
        data.forEach((d, idx) => {
            for (let i = 0; i < Math.min(10, d.count || 5); i++) {
                allPoints.push({x: d.nivel_trafico || idx, y: d['mean'] || 0});
            }
        });
        
        plotData = [{
            x: allPoints.map(p => p.x),
            y: allPoints.map(p => p.y),
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: allPoints.map(p => p.y),
                colorscale: 'Blues',
                showscale: true,
                colorbar: { title: 'ETA (min)' }
            },
            hoverinfo: 'x+y'
        }];
    } else {
        plotData = [{
            x: [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            y: [35, 42, 48, 55, 62, 68, 72, 79, 85, 92],
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: [35, 42, 48, 55, 62, 68, 72, 79, 85, 92],
                colorscale: 'Blues',
                showscale: true,
                colorbar: { title: 'ETA (min)' }
            },
            hoverinfo: 'x+y'
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Nivel de Tráfico (1-5)' },
        yaxis: { title: 'ETA (minutos)' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    renderPlot('chart-trafficImpact', plotData, layout);
}

function createRainImpactChart(data = null) {
    let plotData;
    
    if (data && data.lluvia && data.eta && data.lluvia.length > 0) {
        plotData = [{
            x: data.lluvia,
            y: data.eta,
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: data.eta,
                colorscale: 'Viridis',
                showscale: true,
                colorbar: { title: 'ETA (min)' }
            },
            hoverinfo: 'x+y'
        }];
    } else {
        plotData = [{
            x: [0, 0, 0.5, 0.5, 1.5, 1.5, 3.0, 3.0, 5.0, 5.0],
            y: [42, 48, 52, 58, 65, 72, 78, 85, 92, 98],
            mode: 'markers',
            type: 'scatter',
            marker: {
                size: 10,
                color: [42, 48, 52, 58, 65, 72, 78, 85, 92, 98],
                colorscale: 'Viridis',
                showscale: true,
                colorbar: { title: 'ETA (min)' }
            },
            hoverinfo: 'x+y'
        }];
    }
    
    
    const layout = {
        title: '',
        xaxis: { title: 'Lluvia (mm)' },
        yaxis: { title: 'ETA (minutos)' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    renderPlot('chart-rainImpact', plotData, layout);
}

function createTimeByViaChart(data = null) {
    let plotData;
    
    if (data && Array.isArray(data) && data.length > 0) {
        plotData = data.map((d, idx) => ({
            y: Array(Math.max(5, d.count || 10)).fill(d['mean'] || 0).map((v, i) => v + (Math.random() - 0.5) * 5),
            name: d.tipo_via || `Tipo ${idx}`,
            type: 'box',
            marker: { color: ['#0B3C5D', '#1F6AE1', '#4DA3FF'][idx % 3] }
        }));
    } else {
        plotData = [
            {
                y: [48, 52, 50, 55, 58, 61, 60, 59],
                name: 'Calle',
                type: 'box',
                marker: { color: '#0B3C5D' }
            },
            {
                y: [55, 60, 62, 65, 68, 70, 72, 75],
                name: 'Carrera',
                type: 'box',
                marker: { color: '#1F6AE1' }
            },
            {
                y: [52, 58, 60, 64, 66, 70, 68, 72],
                name: 'Diagonal',
                type: 'box',
                marker: { color: '#4DA3FF' }
            }
        ];
    }
    
    const layout = {
        title: '',
        yaxis: { title: 'Tiempo (minutos)' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: true
    };
    
    renderPlot('chart-timeByVia', plotData, layout);
}

// ========================================================================
// PÁGINA 2: ANÁLISIS FUTURO
// ========================================================================

async function loadAnalisisFuturo() {
    console.log('🔮 Cargando Análisis Futuro...');
    
    try {
        const modelo = appState.filters.model || 'linear';
        const horizon = getNextMonthHorizon();
        const response = await fetch(`${API_BASE_URL}/analytics/future?modelo=${modelo}&month=${horizon}`);
        const result = await response.json();
        const data = result.predicciones || {};

        // Gráfico 1: Demanda y ETA esperado por día
        if (data.demanda_diaria && data.eta_esperado_diario) {
            createRealVsPredictedChart({
                demanda: data.demanda_diaria,
                eta: data.eta_esperado_diario,
                horizon: data.horizon || horizon
            });
        } else {
            createRealVsPredictedChart();
        }

        // Gráfico 2: OTIF y retrasos esperados por día
        if (data.otif_esperado_diario && data.retrasos_esperados_diario) {
            createEtaDistributionChart({
                otif: data.otif_esperado_diario,
                retrasos: data.retrasos_esperados_diario,
                horizon: data.horizon || horizon
            });
        } else {
            createEtaDistributionChart();
        }

        // Gráfico 3: Tráfico esperado vs riesgo de retraso
        if (data.trafico_vs_riesgo && data.trafico_vs_riesgo.length > 0) {
            createRiskByLocalidadChart(data.trafico_vs_riesgo);
        } else if (data.riesgo_por_localidad) {
            createRiskByLocalidadChart(data.riesgo_por_localidad);
        } else {
            createRiskByLocalidadChart();
        }

        // Gráfico 4: Comparativo histórico vs predicho
        if (data.historico_vs_predicho) {
            createHeatmapSensitivityChart(data.historico_vs_predicho);
        } else {
            createHeatmapSensitivityChart();
        }
        
    } catch (error) {
        console.error('Error en Análisis Futuro:', error);
        // Cargar con datos por defecto
        createRealVsPredictedChart();
        createEtaDistributionChart();
        createRiskByLocalidadChart();
        createHeatmapSensitivityChart();
    }
}

function createRealVsPredictedChart(data = null) {
    let plotData;

    if (data && data.demanda && data.eta && data.demanda.y && data.eta.y) {
        plotData = [
            {
                x: data.demanda.x,
                y: data.demanda.y,
                name: 'Pedidos predichos',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#1F6AE1', width: 3 },
                marker: { size: 7 },
                yaxis: 'y1'
            },
            {
                x: data.eta.x,
                y: data.eta.y,
                name: 'ETA esperado',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#27AE60', width: 3 },
                marker: { size: 7 },
                yaxis: 'y2'
            }
        ];
    } else if (data && data.real && data.predicho && data.real.length > 0) {
        plotData = [
            {
                x: data.x || Array.from({length: data.real.length}, (_, i) => i+1),
                y: data.real,
                name: 'ETA Real',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#E74C3C', width: 3 },
                marker: { size: 8 }
            },
            {
                x: data.x || Array.from({length: data.predicho.length}, (_, i) => i+1),
                y: data.predicho,
                name: 'ETA Predicho',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#27AE60', width: 3 },
                marker: { size: 8 }
            }
        ];
    } else {
        plotData = [
            {
                x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                y: [45, 52, 48, 61, 55, 58, 62, 59, 63, 65],
                name: 'ETA Real',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#E74C3C', width: 3 },
                marker: { size: 8 }
            },
            {
                x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                y: [46, 51, 49, 60, 56, 59, 61, 60, 62, 64],
                name: 'ETA Predicho',
                type: 'scatter',
                mode: 'lines+markers',
                line: { color: '#27AE60', width: 3 },
                marker: { size: 8 }
            }
        ];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Día del mes' },
        yaxis: { title: 'Pedidos (n)' },
        yaxis2: {
            title: 'ETA esperado (min)',
            overlaying: 'y',
            side: 'right'
        },
        hovermode: 'x unified',
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: true
    };
    
    renderPlot('chart-realVsPredicted', plotData, layout);
}

function createEtaDistributionChart(errorData = null) {
    let data;

    if (errorData && errorData.otif && errorData.retrasos) {
        data = [
            {
                x: errorData.otif.x,
                y: errorData.otif.y,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'OTIF esperado',
                line: { color: '#27AE60', width: 3 },
                marker: { size: 6 }
            },
            {
                x: errorData.retrasos.x,
                y: errorData.retrasos.y,
                type: 'scatter',
                mode: 'lines+markers',
                name: '% retrasos esperado',
                line: { color: '#E74C3C', width: 3 },
                marker: { size: 6 }
            }
        ];
    } else if (errorData && Array.isArray(errorData) && errorData.length > 0) {
        data = [{
            x: errorData,
            type: 'histogram',
            marker: { color: '#1F6AE1' },
            nbinsx: 30
        }];
    } else {
        data = [{
            x: [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
            type: 'histogram',
            marker: { color: '#1F6AE1' },
            nbinsx: 30
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Día del mes' },
        yaxis: { title: 'Porcentaje (%)', range: [0, 100] },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: true
    };
    
    renderPlot('chart-etaDistribution', data, layout);
}

function createRiskByLocalidadChart(riskData = null) {
    let plotData;

    if (riskData && Array.isArray(riskData) && riskData.length > 0 && riskData[0].traffic !== undefined) {
        plotData = [{
            x: riskData.map(d => d.traffic),
            y: riskData.map(d => d.risk),
            mode: 'markers',
            type: 'scatter',
            text: riskData.map(d => `Día ${d.day}`),
            marker: {
                size: riskData.map(d => Math.max(8, Math.min(28, (d.demand || 40) / 6))),
                color: riskData.map(d => d.risk),
                colorscale: 'Reds',
                showscale: true,
                colorbar: { title: 'Riesgo (%)' }
            },
            hovertemplate: 'Tráfico: %{x:.2f}<br>Riesgo: %{y:.1f}%<br>%{text}<extra></extra>'
        }];
    } else if (riskData && Array.isArray(riskData) && riskData.length > 0) {
        plotData = [{
            x: riskData.map(d => d.localidad),
            y: riskData.map(d => d.riesgo),
            type: 'bar',
            marker: {
                color: riskData.map(d => d.riesgo),
                colorscale: 'Reds'
            }
        }];
    } else {
        plotData = [{
            x: ['Kennedy', 'Ciudad Bolívar', 'Soacha', 'Usaquén', 'Chapinero'],
            y: [65, 72, 58, 48, 55],
            type: 'bar',
            marker: {
                color: [65, 72, 58, 48, 55],
                colorscale: 'Reds'
            }
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: riskData && riskData[0] && riskData[0].traffic !== undefined ? 'Nivel de tráfico esperado' : 'Localidad' },
        yaxis: { title: 'Riesgo de retraso (%)' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    renderPlot('chart-riskByLocalidad', plotData, layout);
}

function createHeatmapSensitivityChart(matrixData = null) {
    let data;

    if (matrixData && Array.isArray(matrixData) && matrixData.length > 0 && matrixData[0].metric) {
        data = [
            {
                x: matrixData.map(d => d.metric),
                y: matrixData.map(d => d.historical),
                type: 'bar',
                name: 'Histórico',
                marker: { color: '#0B3C5D' }
            },
            {
                x: matrixData.map(d => d.metric),
                y: matrixData.map(d => d.predicted),
                type: 'bar',
                name: 'Predicho',
                marker: { color: '#27AE60' }
            }
        ];
    } else if (matrixData && matrixData.z && Array.isArray(matrixData.z)) {
        data = [{
            z: matrixData.z,
            x: matrixData.x || ['0mm', '1mm', '3mm', '5mm'],
            y: matrixData.y || ['Bajo', 'Medio', 'Alto', 'Muy Alto'],
            type: 'heatmap',
            colorscale: 'YlOrRd',
            colorbar: { title: 'ETA (min)' }
        }];
    } else {
        data = [{
            z: [
                [10, 20, 35, 50, 68],
                [15, 25, 40, 55, 72],
                [20, 30, 45, 60, 75],
                [25, 35, 50, 65, 80],
                [30, 40, 55, 70, 85]
            ],
            x: ['Sin lluvia', '0.5mm', '1.5mm', '3mm', '5mm'],
            y: ['Bajo', 'Medio', 'Alto', 'Muy Alto', 'Crítico'],
            type: 'heatmap',
            colorscale: 'YlOrRd',
            colorbar: { title: 'ETA (min)' }
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Métrica' },
        yaxis: { title: 'Valor' },
        barmode: 'group',
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent'
    };
    
    renderPlot('chart-heatmapSensivity', data, layout);
}

// ========================================================================
// PÁGINA 3: CONFIANZA Y MEJORA
// ========================================================================

async function loadConfianzaMejora() {
    console.log('✅ Cargando Confianza y Mejora...');
    
    try {
        const modelo = appState.filters.model || 'linear';
        const response = await fetch(`${API_BASE_URL}/analytics/confidence?modelo=${modelo}`);
        const result = await response.json();
        const data = result.confianza || {};
        
        // Gráfico 1: Comparación de Métricas por Modelo
        if (data.modelos_metricas) {
            renderMetricsComparison({metrics: {}}, data.modelos_metricas);
        } else {
            await loadMetricsComparison();
        }
        
        // Gráfico 2: Evolución del Error
        if (data.evolucion_error) {
            createErrorEvolutionChart(data.evolucion_error);
        } else {
            createErrorEvolutionChart();
        }
        
        // Gráfico 3: Distribución del Error
        if (data.distribucion_error) {
            createErrorDistributionChart(data.distribucion_error);
        } else {
            createErrorDistributionChart();
        }
        
        // Gráfico 4: Propuestas de Mejora
        renderImprovements();
        
    } catch (error) {
        console.error('Error en Confianza y Mejora:', error);
        loadMetricsComparison();
        createErrorEvolutionChart();
        createErrorDistributionChart();
        renderImprovements();
    }
}

async function loadMetricsComparison() {
    try {
        const response = await fetch(`${API_BASE_URL}/metrics/models`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        appState.data.metrics = data;
        
        // Convertir estructura de models a modelos_metricas
        const metricas = data.models.map(m => ({
            modelo: m.model_name,
            mae: m.mae,
            rmse: m.rmse,
            mape: m.mape
        }));
        
        renderMetricsComparison(data, metricas);
        
    } catch (error) {
        console.error('Error cargando métricas:', error);
        showError('No se pudieron cargar las métricas', document.getElementById('chart-metricsComparison'));
    }
}

function renderMetricsComparison(data, metricas = null) {
    let models_list = [];
    let mae_values = [];
    let rmse_values = [];
    let mape_values = [];
    
    if (metricas && Array.isArray(metricas) && metricas.length > 0) {
        models_list = metricas.map(m => m.modelo);
        mae_values = metricas.map(m => m.mae || 0);
        rmse_values = metricas.map(m => m.rmse || 0);
        mape_values = metricas.map(m => m.mape * 100 || 0); // Convertir a porcentaje
    } else if (data.models && Array.isArray(data.models)) {
        models_list = data.models.map(m => m.model_name);
        mae_values = data.models.map(m => m.mae || 0);
        rmse_values = data.models.map(m => m.rmse || 0);
        mape_values = data.models.map(m => m.mape * 100 || 0);
    }
    
    const chartData = [
        {
            x: models_list,
            y: mae_values,
            name: 'MAE',
            type: 'bar',
            marker: { color: '#0B3C5D' }
        },
        {
            x: models_list,
            y: rmse_values,
            name: 'RMSE',
            type: 'bar',
            marker: { color: '#1F6AE1' }
        },
        {
            x: models_list,
            y: mape_values,
            name: 'MAPE (%)',
            type: 'bar',
            marker: { color: '#4DA3FF' }
        }
    ];
    
    const layout = {
        title: '',
        xaxis: { title: 'Modelo' },
        yaxis: { title: 'Valor' },
        barmode: 'group',
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: true
    };
    
    renderPlot('chart-metricsComparison', chartData, layout);
}

function createErrorEvolutionChart(evolutionData = null) {
    let data;
    
    if (evolutionData && Array.isArray(evolutionData) && evolutionData.length > 0) {
        data = [{
            x: Array.from({length: evolutionData.length}, (_, i) => i+1),
            y: evolutionData,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'RMSE',
            line: { color: '#F39C12', width: 3 },
            marker: { size: 8, color: '#F39C12' }
        }];
    } else {
        data = [{
            x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            y: [5.2, 5.1, 5.0, 4.9, 4.95, 4.88, 4.85, 4.82, 4.80, 4.78],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'RMSE',
            line: { color: '#F39C12', width: 3 },
            marker: { size: 8, color: '#F39C12' }
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Período' },
        yaxis: { title: 'RMSE (minutos)' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    renderPlot('chart-errorEvolution', data, layout);
}

function createErrorDistributionChart(errorData = null) {
    let data;
    
    if (errorData && Array.isArray(errorData) && errorData.length > 0) {
        data = [{
            x: errorData,
            type: 'histogram',
            marker: { color: '#E74C3C' },
            nbinsx: 20
        }];
    } else {
        data = [{
            x: [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10],
            type: 'histogram',
            marker: { color: '#E74C3C' },
            nbinsx: 20
        }];
    }
    
    const layout = {
        title: '',
        xaxis: { title: 'Error (Real - Predicho, minutos)' },
        yaxis: { title: 'Frecuencia' },
        plot_bgcolor: 'rgba(244, 246, 249, 0.5)',
        paper_bgcolor: 'transparent',
        showlegend: false
    };
    
    renderPlot('chart-errorDistribution', data, layout);
}

function renderImprovements() {
    const improvements = [
        {
            title: '📦 Plan de capacidad por demanda esperada',
            description: 'Programar cupos de operación y personal por localidad usando la demanda predicha para reducir picos de incumplimiento.'
        },
        {
            title: '🚚 Asignación dinámica de flota',
            description: 'Priorizar motos, vans o bicicletas según ETA esperado y tráfico para mejorar OTIF en franjas críticas.'
        },
        {
            title: '🧭 Estrategia por zonas de riesgo',
            description: 'Definir rutas alternas y ventanas de despacho por zonas con mayor probabilidad de retraso.'
        },
        {
            title: '📲 Comunicación proactiva con clientes',
            description: 'Activar alertas tempranas para pedidos en riesgo y renegociar promesas de entrega antes del incumplimiento.'
        },
        {
            title: '💼 Gestión de SLA y costo logístico',
            description: 'Monitorear OTIF esperado, costo por entrega y nivel de servicio para ajustar operación con foco financiero.'
        }
    ];
    
    const container = document.getElementById('chart-improvements');
    container.innerHTML = '';
    
    improvements.forEach(improvement => {
        const item = document.createElement('div');
        item.className = 'improvement-item';
        item.innerHTML = `
            <div class="improvement-title">${improvement.title}</div>
            <div class="improvement-desc">${improvement.description}</div>
        `;
        container.appendChild(item);
    });
}

// ========================================================================
// GESTIÓN DE FILTROS
// ========================================================================

function initializeFilters() {
    // Eventos de cambio individual
    const filterInputs = document.querySelectorAll('.filter-input, .filter-slider');
    filterInputs.forEach(input => {
        input.addEventListener('change', updateFilterState);
    });
    
    // Actualizar valor del slider
    const trafficSlider = document.getElementById('trafficSlider');
    trafficSlider.addEventListener('input', (e) => {
        document.getElementById('trafficValue').textContent = e.target.value;
        updateFilterState();
    });
    
    // Botón de aplicar filtros
    document.getElementById('applyFiltersBtn').addEventListener('click', applyFilters);
    
    // Botón de limpiar filtros
    document.getElementById('resetFiltersBtn').addEventListener('click', resetFilters);
}

function updateFilterState() {
    appState.filters = {
        localidad: document.getElementById('localidadFilter').value,
        via: document.getElementById('viaFilter').value,
        vehiculo: document.getElementById('vehiculoFilter').value,
        traffic: document.getElementById('trafficSlider').value,
        dateFrom: document.getElementById('dateFrom').value,
        dateTo: document.getElementById('dateTo').value,
        model: document.getElementById('modelSelect').value
    };
}

async function applyFilters() {
    updateFilterState();
    
    console.log('🔄 Aplicando filtros:', appState.filters);
    
    // Aquí se recarga: flashcards, gráficos y datos
    await loadFlashcards();
    await loadPageGraphics();
}

function resetFilters() {
    // Limpiar inputs
    document.getElementById('localidadFilter').value = '';
    document.getElementById('viaFilter').value = '';
    document.getElementById('vehiculoFilter').value = '';
    document.getElementById('trafficSlider').value = '2';
    document.getElementById('dateFrom').value = '';
    document.getElementById('dateTo').value = '';
    document.getElementById('modelSelect').value = '';
    document.getElementById('trafficValue').textContent = '2';
    
    // Resetear estado
    appState.filters = {
        localidad: '',
        via: '',
        vehiculo: '',
        traffic: 2,
        dateFrom: null,
        dateTo: null,
        model: ''
    };
    
    console.log('✨ Filtros limpiados');
    
    // Recargar datos
    loadFlashcards();
    loadPageGraphics();
}

// ========================================================================
// UTILIDADES DE UI
// ========================================================================

function showLoading(container) {
    container.innerHTML = `
        <div class="loading">
            <div class="spinner"></div>
        </div>
    `;
}

function showError(message, container) {
    container.innerHTML = `
        <div class="error-message">
            ⚠️ ${message}. Por favor, intenta más tarde.
        </div>
    `;
}

// ========================================================================
// UTILIDADES DE API
// ========================================================================

/**
 * Realiza una llamada fetch con manejo de errores
 * @param {string} url - URL del endpoint
 * @param {object} options - Opciones de fetch
 * @returns {Promise} Respuesta JSON
 */
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error(`API Call Error (${url}):`, error);
        throw error;
    }
}

// ========================================================================
// LOG DE INICIALIZACIÓN
// ========================================================================

console.log('%c🎯 Dashboard Intrak Cargado', 'color: #0B3C5D; font-size: 16px; font-weight: bold;');
console.log('%cAPI Base: ' + API_BASE_URL, 'color: #1F6AE1;');
