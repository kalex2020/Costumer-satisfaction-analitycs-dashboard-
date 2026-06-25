#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Servidor HTTP simple para servir el frontend del dashboard Intrak.
Escucha en http://localhost:3000
"""

import http.server
import socketserver
import os
from pathlib import Path
from urllib.parse import urlsplit

PORT = 3000
FRONTEND_DIR = Path(__file__).parent.absolute()

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)
    
    def end_headers(self):
        """Agregar CORS headers para permitir llamadas a API"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store')
        super().end_headers()
    
    def do_GET(self):
        """Manejar rutas: servir index.html para rutas no encontradas"""
        request_path = urlsplit(self.path).path

        if request_path == '/' or not os.path.exists(FRONTEND_DIR / request_path.lstrip('/')):
            self.path = '/index.html'
        else:
            self.path = request_path

        return super().do_GET()

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"\n{'='*70}")
        print(f"🌐 Servidor Frontend Intrak")
        print(f"{'='*70}")
        print(f"✅ Servidor escuchando en: http://localhost:{PORT}")
        print(f"📁 Sirviendo desde: {FRONTEND_DIR}")
        print(f"🔌 Backend API: http://localhost:8000")
        print(f"\n⚠️  Presiona CTRL+C para detener el servidor\n")
        print(f"{'='*70}\n")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n✋ Servidor detenido")
