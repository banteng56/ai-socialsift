<template>
  <section id="us">
    <v-container fluid>
      <v-row align="center" justify="center">
        <v-col cols="10">
          <v-row align="center" justify="center">
            <v-col sm="4" class="hidden-xs-only">
              <v-img src="@/assets/img/logo.png" class="d-block ml-auto mr-auto" max-width="350px" />
            </v-col>
            <v-col cols="12" sm="8" class="white--text text-center">
              
              <v-row justify="center">
                <v-col cols="12" sm="4">
                  <p class="mb-2"><strong>Kristian Bintang</strong></p>
                  <v-btn rounded outlined href="https://github.com/" target="_blank" large color="white">
                    <v-icon class="mr-2">mdi-github</v-icon>GitHub
                  </v-btn>
                </v-col>
                <v-col cols="12" sm="4">
                  <p class="mb-2"><strong>Muhammad Surya</strong></p>
                  <v-btn rounded outlined href="https://github.com/" target="_blank" large color="white">
                    <v-icon class="mr-2">mdi-github</v-icon>GitHub
                  </v-btn>
                </v-col>
                <v-col cols="12" sm="4">
                  <p class="mb-2"><strong>Louisa</strong></p>
                  <v-btn rounded outlined href="https://github.com/" target="_blank" large color="white">
                    <v-icon class="mr-2">mdi-github</v-icon>GitHub
                  </v-btn>
                </v-col>
              </v-row>
            </v-col>
          </v-row>
        </v-col>
      </v-row>
    </v-container>
  </section>
</template>



<style scoped>
#us {
  background-image: url("~@/assets/img/bgDownload.jpg");
  background-attachment: fixed;
  background-repeat: no-repeat;
  background-size: cover;
  background-position: center;
  height: 500px;
}

#us .container,
#us .row {
  height: 100%;
}
</style>