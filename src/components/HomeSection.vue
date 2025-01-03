<template>
  <section id="hero">
    <v-parallax dark src="@/assets/img/bghero1.jpg" height="750">
      <v-row align="center" justify="center">
        <v-col cols="10">
          <v-row align="center" justify="center">
            <v-col cols="12" md="6" xl="8">
              <h1 class="display-2 font-weight-bold mb-4">SocialSift</h1>
              <h1 class="font-weight-light">
                Pahami Percakapan dan Temukan Tren <br />
                Ambil Tindakan Cerdas di Dunia Digital <br />
              </h1>
              <v-btn
                rounded
                outlined
                large
                dark
                @click="fileDialog = true"
                class="mt-5"
              >
                Start
                <v-icon class="ml-2">mdi-arrow-down</v-icon>
              </v-btn>
              <v-btn
  rounded
  outlined
  large
  dark
  class="mt-5"
  to="/analisis"
>
  Analisis
  <v-icon class="ml-2">mdi-chart-bar</v-icon>
</v-btn>

            </v-col>
            <v-col cols="12" md="6" xl="4" class="hidden-sm-and-down"> </v-col>
          </v-row>
        </v-col>
      </v-row>
      <div class="svg-border-waves text-white">
        <v-img src="@/assets/img/borderWaves.svg" />
      </div>
    </v-parallax>

    <v-dialog v-model="fileDialog" max-width="500px">
      <v-card>
        <v-card-title class="headline">Upload File CSV</v-card-title>
        <v-card-text>
          <v-file-input
            v-model="file"
            label="Choose a CSV file"
            accept=".csv"
            outlined
            dense
            required
          />
        </v-card-text>
        <v-card-actions>
          <v-btn text @click="fileDialog = false">Cancel</v-btn>
          <v-btn color="primary" @click="uploadFile">Upload</v-btn>
        </v-card-actions>
      </v-card>
    </v-dialog>

    <!-- Loading Spinner ketika file sedang diproses -->
    <v-dialog v-model="isLoading" persistent max-width="400px">
      <v-card>
        <v-card-title class="headline">Processing...</v-card-title>
        <v-card-text class="text-center">
          <v-progress-circular indeterminate color="primary" />
          <p>Please wait while we process the file.</p>
        </v-card-text>
      </v-card>
    </v-dialog>

    <!-- Section Fitur -->
    <v-container fluid id="features" class="mt-2">
      <v-row align="center" justify="center">
        <v-col cols="10">
          <v-row align="center" justify="space-around">
            <v-col
              cols="12"
              sm="4"
              class="text-center"
              v-for="(feature, i) in features"
              :key="i"
            >
              <v-hover v-slot:default="{ hover }">
                <v-card
                  class="card"
                  shaped
                  :elevation="hover ? 10 : 4"
                  :class="{ up: hover }"
                >
                  <v-img
                    :src="feature.img"
                    max-width="100px"
                    class="d-block ml-auto mr-auto"
                    :class="{ 'zoom-efect': hover }"
                  ></v-img>
                  <h1 class="font-weight-regular">{{ feature.title }}</h1>
                  <h4 class="font-weight-regular subtitle-1">
                    {{ feature.text }}
                  </h4>
                </v-card>
              </v-hover>
            </v-col>
          </v-row>
        </v-col>
      </v-row>
    </v-container>

    <div class="svg-border-waves">
      <img src="~@/assets/img/wave2.svg" />
    </div>
  </section>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      features: [
        {
          img: require("@/assets/img/icon3.png"),
          title: "Classification",
          text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
        },
        {
          img: require("@/assets/img/icon1.png"),
          title: "Sentiment",
          text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
        },
        {
          img: require("@/assets/img/icon2.png"),
          title: "Entity",
          text: "Lorem ipsum dolor sit amet consectetur adipisicing elit.",
        },
      ],
      fileDialog: false, // untuk kontrol dialog
      file: null, // menampung file yang dipilih
      isLoading: false, // untuk kontrol spinner loading
    };
  },
  methods: {
    // Fungsi untuk menangani klik tombol upload
    async uploadFile() {
      if (!this.file) {
        alert("Please select a file first.");
        return;
      }

      this.isLoading = true; // Menampilkan spinner loading

      const formData = new FormData();
      formData.append("file", this.file);

      try {
        // Kirim file ke API
        const response = await axios.post("https://backend.socialsift.biz.id/upload-csv/", formData, {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        });

        // Proses hasil dari API dan tampilkan ke UI
        console.log(response.data.processed_data);
        
        this.fileDialog = false; // Menutup dialog setelah berhasil
        this.isLoading = false; // Menyembunyikan spinner loading

        // Di sini bisa dilakukan hal lain setelah sukses upload
      } catch (error) {
        console.error("Error uploading file:", error);
        this.isLoading = false; // Menyembunyikan spinner loading jika error
        alert("An error occurred while uploading the file.");
      }
    },
  },
};
</script>

<style lang="scss">
.circle {
  stroke: white;
  stroke-dasharray: 650;
  stroke-dashoffset: 650;
  transition: all 0.5s ease-in-out;
  opacity: 0.3;
}

.playBut {
  display: inline-block;
  transition: all 0.5s ease;

  .triangle {
    transition: all 0.7s ease-in-out;
    stroke-dasharray: 240;
    stroke-dashoffset: 480;
    stroke: white;
    transform: translateY(0);
  }

  &:hover {
    .triangle {
      stroke-dashoffset: 0;
      opacity: 1;
      stroke: white;
      animation: nudge 0.7s ease-in-out;

      @keyframes nudge {
        0% {
          transform: translateX(0);
        }
        30% {
          transform: translateX(-5px);
        }
        50% {
          transform: translateX(5px);
        }
        70% {
          transform: translateX(-2px);
        }
        100% {
          transform: translateX(0);
        }
      }
    }

    .circle {
      stroke-dashoffset: 0;
      opacity: 1;
    }
  }
}
</style>

<style>
.btn-play {
  transition: 0.2s;
}

.svg-border-waves .v-image {
  position: absolute;
  bottom: 0;
  left: 0;
  height: 3rem;
  width: 100%;
  overflow: hidden;
}

#hero {
  z-index: 0;
}
.svg-border-waves img {
  position: absolute;
  bottom: 0;
  left: 0;
  width: 100%;
  margin-bottom: -2px;
  z-index: -1;
}

.card {
  min-height: 300px;
  padding: 10px;
  transition: 0.5s ease-out;
}

.card .v-image {
  margin-bottom: 15px;
  transition: 0.75s;
}

.card h1 {
  margin-bottom: 10px;
}

.zoom-efect {
  transform: scale(1.1);
}

.up {
  transform: translateY(-20px);
  transition: 0.5s ease-out;
}
</style>

<style>
section {
  position: relative;
}
</style>
