window.global ||= window;

import './assets/main.css'

import { createApp } from 'vue'
import App from './App.vue'

// Vuetify
import 'vuetify/styles'
import '@mdi/font/css/materialdesignicons.css' // Ensure you are using css-loader
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

import 'highlight.js/styles/stackoverflow-dark.css'
import 'highlight.js/lib/common';
import hljsVuePlugin from "@highlightjs/vue-plugin";



const vuetify = createVuetify({
  components,
  directives,
})

createApp(App).use(vuetify).use(hljsVuePlugin).mount('#app')