import Vue from 'vue';
import App from './App.vue';
import vuetify from './plugins/vuetify';
import VueYouTubeEmbed from 'vue-youtube-embed';
import router from './router'; 

Vue.use(VueYouTubeEmbed);
Vue.config.productionTip = false;

new Vue({
  vuetify,
  router, 
  render: h => h(App),
}).$mount('#app');
