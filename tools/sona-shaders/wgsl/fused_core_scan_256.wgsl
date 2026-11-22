struct type_7_ {
    member: array<vec4<f32>>,
}

struct type_16_ {
    member: u32,
    member_1_: u32,
}

struct type_17_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: u32,
    member_5_: f32,
    member_6_: f32,
    member_7_: u32,
}

struct type_18_ {
    member: type_17_,
}

struct type_24_ {
    member: array<f32, 4>,
}

struct type_26_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_27_ {
    member: type_24_,
    member_1_: type_24_,
}

struct type_32_ {
    member: type_27_,
    member_1_: type_27_,
    member_2_: type_27_,
    member_3_: type_27_,
}

struct type_33_ {
    member: type_24_,
    member_1_: type_24_,
    member_2_: type_24_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> num_wg_1_: vec3<u32>;
@group(0) @binding(0) 
var<uniform> uniforms: type_18_;
@group(0) @binding(1) 
var<storage> core_inputs: type_7_;
@group(0) @binding(2) 
var<storage> quad_params: type_7_;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(3) 
var<storage, read_write> tile_checkpoints: type_7_;
@group(0) @binding(4) 
var<storage, read_write> feature_outputs: type_7_;
var<workgroup> lane_buf: array<vec4<f32>, 512>;
var<workgroup> aux_scratch: array<f32, 256>;

fn function_() {
    var local: array<f32, 4>;
    var local_1_: array<f32, 4>;
    var local_2_: array<f32, 4>;
    var local_3_: array<f32, 4>;
    var local_4_: array<f32, 4>;
    var local_5_: array<f32, 4>;
    var local_6_: array<f32, 4>;
    var local_7_: array<f32, 4>;
    var local_8_: array<f32, 4>;
    var local_9_: array<f32, 4>;
    var local_10_: array<f32, 4>;
    var local_11_: array<f32, 4>;
    var local_12_: array<f32, 4>;
    var local_13_: array<f32, 4>;
    var local_14_: array<f32, 4>;
    var local_15_: array<f32, 4>;
    var phi_233_: u32;
    var phi_264_: u32;
    var phi_275_: u32;
    var phi_278_: u32;
    var phi_276_: u32;
    var phi_279_: u32;
    var phi_290_: u32;
    var phi_293_: u32;
    var phi_295_: u32;
    var phi_297_: u32;
    var phi_311_: u32;
    var phi_312_: u32;
    var phi_291_: u32;
    var phi_294_: u32;
    var phi_296_: u32;
    var phi_298_: u32;
    var local_16_: u32;
    var phi_316_: u32;
    var phi_317_: u32;
    var phi_331_: u32;
    var phi_332_: u32;
    var phi_349_: vec4<f32>;
    var phi_362_: vec4<f32>;
    var phi_375_: vec4<f32>;
    var phi_388_: vec4<f32>;
    var phi_401_: vec4<f32>;
    var phi_414_: vec4<f32>;
    var phi_427_: vec4<f32>;
    var phi_440_: vec4<f32>;
    var phi_453_: vec4<f32>;
    var phi_466_: vec4<f32>;
    var phi_479_: vec4<f32>;
    var phi_492_: vec4<f32>;
    var phi_505_: vec4<f32>;
    var phi_518_: vec4<f32>;
    var phi_531_: vec4<f32>;
    var phi_544_: vec4<f32>;
    var phi_557_: vec4<f32>;
    var phi_570_: vec4<f32>;
    var phi_583_: vec4<f32>;
    var phi_596_: vec4<f32>;
    var phi_609_: vec4<f32>;
    var phi_622_: vec4<f32>;
    var phi_635_: vec4<f32>;
    var phi_661_: vec4<f32>;
    var phi_688_: bool;
    var phi_694_: f32;
    var phi_695_: f32;
    var phi_702_: f32;
    var phi_728_: bool;
    var phi_734_: f32;
    var phi_735_: f32;
    var phi_742_: f32;
    var phi_768_: bool;
    var phi_774_: f32;
    var phi_775_: f32;
    var phi_782_: f32;
    var phi_808_: bool;
    var phi_814_: f32;
    var phi_815_: f32;
    var phi_822_: f32;
    var phi_854_: vec4<f32>;
    var phi_866_: bool;
    var phi_883_: bool;
    var phi_900_: bool;
    var phi_917_: bool;
    var phi_936_: vec4<f32>;
    var phi_948_: bool;
    var phi_965_: bool;
    var phi_982_: bool;
    var phi_999_: bool;
    var phi_1018_: vec4<f32>;
    var phi_1030_: bool;
    var phi_1047_: bool;
    var phi_1064_: bool;
    var phi_1081_: bool;
    var phi_1100_: vec4<f32>;
    var phi_1112_: bool;
    var phi_1129_: bool;
    var phi_1146_: bool;
    var phi_1163_: bool;
    var phi_1182_: vec4<f32>;
    var phi_1194_: bool;
    var phi_1211_: bool;
    var phi_1228_: bool;
    var phi_1245_: bool;
    var phi_1264_: vec4<f32>;
    var phi_1276_: bool;
    var phi_1293_: bool;
    var phi_1310_: bool;
    var phi_1327_: bool;
    var phi_1346_: vec4<f32>;
    var phi_1358_: bool;
    var phi_1375_: f32;
    var phi_1387_: bool;
    var phi_1404_: f32;
    var phi_1416_: bool;
    var phi_1433_: f32;
    var phi_1445_: bool;
    var phi_1462_: f32;
    var phi_1476_: vec4<f32>;
    var phi_1488_: bool;
    var phi_1505_: f32;
    var phi_1517_: bool;
    var phi_1534_: f32;
    var phi_1546_: bool;
    var phi_1563_: f32;
    var phi_1575_: bool;
    var phi_1592_: f32;
    var phi_1603_: type_32_;
    var phi_1605_: type_32_;
    var phi_1606_: bool;
    var phi_1609_: bool;
    var phi_1610_: type_32_;
    var phi_1611_: bool;
    var phi_1613_: bool;
    var phi_1614_: bool;
    var phi_1636_: type_32_;
    var phi_17856_: bool;
    var phi_1640_: type_16_;
    var phi_1643_: type_32_;
    var phi_1641_: type_16_;
    var phi_1666_: type_16_;
    var phi_1680_: u32;
    var phi_1700_: bool;
    var phi_1749_: bool;
    var phi_1765_: bool;
    var phi_1781_: bool;
    var phi_1797_: bool;
    var phi_1829_: bool;
    var phi_1845_: bool;
    var phi_1861_: bool;
    var phi_1877_: bool;
    var phi_1909_: bool;
    var phi_1925_: bool;
    var phi_1941_: bool;
    var phi_1957_: bool;
    var phi_1989_: bool;
    var phi_2005_: bool;
    var phi_2021_: bool;
    var phi_2037_: bool;
    var phi_2069_: bool;
    var phi_2085_: bool;
    var phi_2101_: bool;
    var phi_2117_: bool;
    var phi_2149_: bool;
    var phi_2165_: bool;
    var phi_2181_: bool;
    var phi_2197_: bool;
    var phi_2229_: bool;
    var phi_2245_: bool;
    var phi_2261_: bool;
    var phi_2277_: bool;
    var phi_2309_: bool;
    var phi_2325_: bool;
    var phi_2341_: bool;
    var phi_2357_: bool;
    var phi_2404_: vec4<f32>;
    var phi_2417_: vec4<f32>;
    var phi_2418_: vec4<f32>;
    var phi_2419_: vec4<f32>;
    var phi_17850_: bool;
    var phi_2436_: type_16_;
    var phi_2439_: type_32_;
    var phi_2437_: type_16_;
    var phi_2462_: type_16_;
    var phi_2480_: f32;
    var phi_2491_: f32;
    var phi_2502_: f32;
    var phi_2513_: f32;
    var phi_2524_: f32;
    var phi_2535_: f32;
    var phi_2557_: f32;
    var phi_2568_: f32;
    var phi_2579_: f32;
    var phi_2590_: f32;
    var phi_2601_: f32;
    var phi_2612_: f32;
    var phi_2623_: f32;
    var phi_2634_: f32;
    var phi_2645_: f32;
    var phi_2656_: f32;
    var phi_2681_: bool;
    var phi_2687_: f32;
    var phi_2688_: f32;
    var phi_2695_: f32;
    var phi_2720_: bool;
    var phi_2726_: f32;
    var phi_2727_: f32;
    var phi_2734_: f32;
    var phi_2759_: bool;
    var phi_2765_: f32;
    var phi_2766_: f32;
    var phi_2773_: f32;
    var phi_2798_: bool;
    var phi_2804_: f32;
    var phi_2805_: f32;
    var phi_2812_: f32;
    var phi_2837_: bool;
    var phi_2843_: f32;
    var phi_2844_: f32;
    var phi_2851_: f32;
    var phi_2876_: bool;
    var phi_2882_: f32;
    var phi_2883_: f32;
    var phi_2890_: f32;
    var phi_2961_: f32;
    var phi_2972_: f32;
    var phi_2983_: f32;
    var phi_2994_: f32;
    var phi_3005_: f32;
    var phi_3056_: f32;
    var phi_3070_: f32;
    var phi_3081_: bool;
    var phi_3087_: f32;
    var phi_3101_: f32;
    var phi_3112_: bool;
    var phi_3118_: f32;
    var phi_4000_: type_26_;
    var phi_4078_: u32;
    var phi_4081_: type_26_;
    var phi_4177_: type_26_;
    var phi_4079_: u32;
    var phi_4082_: type_26_;
    var phi_17809_: bool;
    var phi_4289_: array<f32, 2>;
    var phi_4301_: u32;
    var phi_4359_: array<f32, 2>;
    var phi_4620_: type_26_;
    var phi_4698_: u32;
    var phi_4701_: type_26_;
    var phi_4797_: type_26_;
    var phi_4699_: u32;
    var phi_4702_: type_26_;
    var phi_18124_: bool;
    var phi_4909_: array<f32, 2>;
    var phi_4921_: u32;
    var phi_4979_: array<f32, 2>;
    var phi_5229_: f32;
    var phi_3168_: type_16_;
    var phi_3171_: f32;
    var phi_3173_: f32;
    var phi_3169_: type_16_;
    var phi_3196_: type_16_;
    var phi_12029_: bool;
    var phi_12089_: bool;
    var phi_12149_: bool;
    var phi_12209_: bool;
    var phi_3363_: u32;
    var phi_3364_: u32;
    var phi_3415_: u32;
    var phi_3416_: u32;
    var phi_3172_: f32;
    var phi_3174_: f32;
    var phi_18133_: bool;
    var local_17_: f32;
    var local_18_: f32;
    var phi_18177_: bool;
    var phi_3538_: type_16_;
    var phi_3541_: f32;
    var phi_3543_: f32;
    var phi_3539_: type_16_;
    var phi_3566_: type_16_;
    var phi_12269_: bool;
    var phi_12329_: bool;
    var phi_12389_: bool;
    var phi_12449_: bool;
    var phi_3733_: u32;
    var phi_3734_: u32;
    var phi_3785_: u32;
    var phi_3786_: u32;
    var phi_3542_: f32;
    var phi_3544_: f32;
    var phi_18173_: bool;
    var local_19_: f32;
    var local_20_: f32;
    var phi_18225_: bool;
    var phi_3950_: f32;
    var phi_3962_: f32;
    var phi_18218_: bool;
    var phi_5271_: f32;
    var phi_5272_: f32;
    var phi_5324_: type_16_;
    var phi_5327_: f32;
    var phi_5329_: f32;
    var phi_5325_: type_16_;
    var phi_5352_: type_16_;
    var phi_5405_: u32;
    var phi_5406_: u32;
    var phi_12509_: bool;
    var phi_12569_: bool;
    var phi_12629_: bool;
    var phi_12689_: bool;
    var phi_12749_: bool;
    var phi_12809_: bool;
    var phi_12869_: bool;
    var phi_12929_: bool;
    var phi_6147_: u32;
    var phi_6148_: u32;
    var phi_6209_: u32;
    var phi_6210_: u32;
    var phi_13109_: bool;
    var phi_13169_: bool;
    var phi_13229_: bool;
    var phi_13289_: bool;
    var phi_13349_: bool;
    var phi_13409_: bool;
    var phi_13469_: bool;
    var phi_13529_: bool;
    var phi_6951_: u32;
    var phi_6952_: u32;
    var phi_5328_: f32;
    var phi_5330_: f32;
    var phi_18214_: bool;
    var local_21_: f32;
    var local_22_: f32;
    var phi_18510_: bool;
    var phi_7094_: type_32_;
    var phi_7106_: f32;
    var phi_7118_: f32;
    var phi_13709_: bool;
    var phi_13724_: bool;
    var phi_7156_: bool;
    var phi_7162_: f32;
    var phi_7180_: bool;
    var phi_7186_: f32;
    var phi_18499_: bool;
    var phi_2440_: type_32_;
    var phi_18498_: bool;
    var phi_7239_: u32;
    var phi_7240_: u32;
    var phi_7254_: u32;
    var phi_7255_: u32;
    var phi_7269_: u32;
    var phi_7270_: u32;
    var phi_7284_: u32;
    var phi_7285_: u32;
    var phi_7307_: u32;
    var phi_7308_: u32;
    var phi_7322_: u32;
    var phi_7323_: u32;
    var phi_7337_: u32;
    var phi_7338_: u32;
    var phi_7352_: u32;
    var phi_7353_: u32;
    var local_23_: type_32_;
    var phi_7378_: u32;
    var phi_7379_: u32;
    var phi_7393_: u32;
    var phi_7394_: u32;
    var phi_7408_: u32;
    var phi_7409_: u32;
    var phi_7423_: u32;
    var phi_7424_: u32;
    var phi_7446_: u32;
    var phi_7447_: u32;
    var phi_7461_: u32;
    var phi_7462_: u32;
    var phi_7476_: u32;
    var phi_7477_: u32;
    var phi_7491_: u32;
    var phi_7492_: u32;
    var local_24_: type_32_;
    var phi_7563_: type_24_;
    var phi_7564_: type_24_;
    var phi_13739_: bool;
    var phi_13754_: bool;
    var phi_13769_: bool;
    var phi_13784_: bool;
    var phi_13799_: bool;
    var phi_13814_: bool;
    var phi_13829_: bool;
    var phi_13844_: bool;
    var phi_13859_: bool;
    var phi_13874_: bool;
    var phi_13889_: bool;
    var phi_13904_: bool;
    var phi_13919_: bool;
    var phi_13934_: bool;
    var phi_13949_: bool;
    var phi_13964_: bool;
    var phi_13979_: bool;
    var phi_13994_: bool;
    var phi_14009_: bool;
    var phi_14024_: bool;
    var phi_14039_: bool;
    var phi_14054_: bool;
    var phi_14069_: bool;
    var phi_14084_: bool;
    var phi_14099_: bool;
    var phi_14114_: bool;
    var phi_14129_: bool;
    var phi_14144_: bool;
    var phi_14159_: bool;
    var phi_14174_: bool;
    var phi_14189_: bool;
    var phi_14204_: bool;
    var phi_14219_: bool;
    var phi_14234_: bool;
    var phi_14249_: bool;
    var phi_14264_: bool;
    var phi_14279_: bool;
    var phi_14294_: bool;
    var phi_14309_: bool;
    var phi_14324_: bool;
    var phi_14339_: bool;
    var phi_14354_: bool;
    var phi_14369_: bool;
    var phi_14384_: bool;
    var phi_14399_: bool;
    var phi_14414_: bool;
    var phi_14429_: bool;
    var phi_14444_: bool;
    var phi_14459_: bool;
    var phi_14474_: bool;
    var phi_14489_: bool;
    var phi_14504_: bool;
    var phi_14519_: bool;
    var phi_14534_: bool;
    var phi_14549_: bool;
    var phi_14564_: bool;
    var phi_14579_: bool;
    var phi_14594_: bool;
    var phi_14609_: bool;
    var phi_14624_: bool;
    var phi_14639_: bool;
    var phi_14654_: bool;
    var phi_14669_: bool;
    var phi_14684_: bool;
    var phi_7999_: bool;
    var phi_8001_: bool;
    var phi_8002_: bool;
    var phi_8003_: bool;
    var phi_8004_: bool;
    var phi_14699_: bool;
    var phi_14714_: bool;
    var phi_14729_: bool;
    var phi_14744_: bool;
    var phi_8382_: type_33_;
    var phi_14759_: bool;
    var phi_14774_: bool;
    var phi_14789_: bool;
    var phi_14804_: bool;
    var phi_14819_: bool;
    var phi_14834_: bool;
    var phi_14849_: bool;
    var phi_14864_: bool;
    var phi_14879_: bool;
    var phi_14894_: bool;
    var phi_14909_: bool;
    var phi_14924_: bool;
    var phi_14939_: bool;
    var phi_14954_: bool;
    var phi_14969_: bool;
    var phi_14984_: bool;
    var phi_14999_: bool;
    var phi_15014_: bool;
    var phi_15029_: bool;
    var phi_15044_: bool;
    var phi_15059_: bool;
    var phi_15074_: bool;
    var phi_15089_: bool;
    var phi_15104_: bool;
    var phi_15119_: bool;
    var phi_15134_: bool;
    var phi_15149_: bool;
    var phi_15164_: bool;
    var phi_15179_: bool;
    var phi_15194_: bool;
    var phi_15209_: bool;
    var phi_15224_: bool;
    var phi_15239_: bool;
    var phi_15254_: bool;
    var phi_15269_: bool;
    var phi_15284_: bool;
    var phi_15299_: bool;
    var phi_15314_: bool;
    var phi_15329_: bool;
    var phi_15344_: bool;
    var phi_15359_: bool;
    var phi_15374_: bool;
    var phi_15389_: bool;
    var phi_15404_: bool;
    var phi_15419_: bool;
    var phi_15434_: bool;
    var phi_15449_: bool;
    var phi_15464_: bool;
    var phi_15479_: bool;
    var phi_15494_: bool;
    var phi_15509_: bool;
    var phi_15524_: bool;
    var phi_15539_: bool;
    var phi_15554_: bool;
    var phi_15569_: bool;
    var phi_15584_: bool;
    var phi_15599_: bool;
    var phi_15614_: bool;
    var phi_15629_: bool;
    var phi_15644_: bool;
    var phi_15659_: bool;
    var phi_15674_: bool;
    var phi_15689_: bool;
    var phi_15704_: bool;
    var phi_8818_: bool;
    var phi_8820_: bool;
    var phi_8821_: bool;
    var phi_8822_: bool;
    var phi_8823_: bool;
    var phi_15719_: bool;
    var phi_15734_: bool;
    var phi_15749_: bool;
    var phi_15764_: bool;
    var phi_9201_: type_33_;
    var phi_9218_: bool;
    var phi_9220_: bool;
    var phi_9221_: bool;
    var phi_9222_: bool;
    var phi_9223_: bool;
    var phi_15779_: bool;
    var phi_15794_: bool;
    var phi_15809_: bool;
    var phi_15824_: bool;
    var phi_15839_: bool;
    var phi_15854_: bool;
    var phi_15869_: bool;
    var phi_15884_: bool;
    var phi_15899_: bool;
    var phi_15914_: bool;
    var phi_15929_: bool;
    var phi_15944_: bool;
    var phi_15959_: bool;
    var phi_15974_: bool;
    var phi_15989_: bool;
    var phi_16004_: bool;
    var phi_16019_: bool;
    var phi_16034_: bool;
    var phi_16049_: bool;
    var phi_16064_: bool;
    var phi_16079_: bool;
    var phi_16094_: bool;
    var phi_16109_: bool;
    var phi_16124_: bool;
    var phi_16139_: bool;
    var phi_16154_: bool;
    var phi_16169_: bool;
    var phi_16184_: bool;
    var phi_16199_: bool;
    var phi_16214_: bool;
    var phi_16229_: bool;
    var phi_16244_: bool;
    var phi_16259_: bool;
    var phi_16274_: bool;
    var phi_16289_: bool;
    var phi_16304_: bool;
    var phi_16319_: bool;
    var phi_16334_: bool;
    var phi_16349_: bool;
    var phi_16364_: bool;
    var phi_16379_: bool;
    var phi_16394_: bool;
    var phi_16409_: bool;
    var phi_16424_: bool;
    var phi_16679_: bool;
    var phi_16694_: bool;
    var phi_16709_: bool;
    var phi_16724_: bool;
    var phi_16739_: bool;
    var phi_16754_: bool;
    var phi_16769_: bool;
    var phi_16784_: bool;
    var phi_16799_: bool;
    var phi_16814_: bool;
    var phi_16829_: bool;
    var phi_16844_: bool;
    var phi_16859_: bool;
    var phi_16874_: bool;
    var phi_16889_: bool;
    var phi_16904_: bool;
    var phi_16919_: bool;
    var phi_16934_: bool;
    var phi_16949_: bool;
    var phi_16964_: bool;
    var phi_16979_: bool;
    var phi_16994_: bool;
    var phi_17009_: bool;
    var phi_17024_: bool;
    var phi_17039_: bool;
    var phi_17054_: bool;
    var phi_17069_: bool;
    var phi_17084_: bool;
    var phi_17099_: bool;
    var phi_17114_: bool;
    var phi_17129_: bool;
    var phi_17144_: bool;
    var phi_17159_: bool;
    var phi_17174_: bool;
    var phi_17189_: bool;
    var phi_17204_: bool;
    var phi_17219_: bool;
    var phi_17234_: bool;
    var phi_17249_: bool;
    var phi_17264_: bool;
    var phi_17279_: bool;
    var phi_17294_: bool;
    var phi_17309_: bool;
    var phi_17324_: bool;
    var phi_17339_: bool;
    var phi_17354_: bool;
    var phi_17369_: bool;
    var phi_17384_: bool;
    var phi_17399_: bool;
    var phi_17414_: bool;
    var phi_17429_: bool;
    var phi_17444_: bool;
    var phi_17699_: bool;
    var phi_17714_: bool;
    var phi_17729_: bool;
    var phi_17744_: bool;
    var phi_17759_: bool;
    var phi_17774_: bool;
    var phi_17789_: bool;
    var phi_17804_: bool;
    var phi_11111_: type_24_;
    var phi_11112_: type_24_;
    var phi_11271_: bool;
    var phi_11287_: bool;
    var phi_11303_: bool;
    var phi_11319_: bool;
    var phi_11357_: bool;
    var phi_11373_: bool;
    var phi_11389_: bool;
    var phi_11405_: bool;
    var phi_11437_: bool;
    var phi_11453_: bool;
    var phi_11469_: bool;
    var phi_11485_: bool;
    var phi_11517_: bool;
    var phi_11533_: bool;
    var phi_11549_: bool;
    var phi_11565_: bool;
    var phi_11597_: bool;
    var phi_11613_: bool;
    var phi_11629_: bool;
    var phi_11645_: bool;
    var phi_11677_: bool;
    var phi_11693_: bool;
    var phi_11709_: bool;
    var phi_11725_: bool;
    var phi_11766_: bool;
    var phi_11783_: f32;
    var phi_11795_: bool;
    var phi_11812_: f32;
    var phi_11824_: bool;
    var phi_11841_: f32;
    var phi_11853_: bool;
    var phi_11870_: f32;
    var phi_11891_: bool;
    var phi_11908_: f32;
    var phi_11920_: bool;
    var phi_11937_: f32;
    var phi_11949_: bool;
    var phi_11966_: f32;
    var phi_11978_: bool;
    var phi_11995_: f32;
    var phi_12000_: type_32_;
    var phi_18516_: bool;
    var phi_1644_: type_32_;
    var phi_18515_: bool;
    var local_25_: u32;
    var local_26_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e109_ = wg_id_1_;
            let _e110_ = num_wg_1_;
            let _e111_ = lid_1_;
            let _e113_ = arrayLength((&core_inputs.member));
            let _e115_ = arrayLength((&quad_params.member));
            let _e117_ = arrayLength((&tile_checkpoints.member));
            let _e119_ = arrayLength((&feature_outputs.member));
            let _e120_ = (_e111_ & 255u);
            let _e128_ = select(1u, _e110_.y, (1u < _e110_.y));
            let _e131_ = ((_e109_.z * select(1u, _e110_.x, (1u < _e110_.x))) + _e109_.x);
            let _e134_ = uniforms.member.member;
            if (_e134_ > 0u) {
                phi_233_ = (((_e134_ - 1u) >> bitcast<u32>(8i)) + 1u);
            } else {
                phi_233_ = 0u;
            }
            let _e141_ = phi_233_;
            let _e144_ = uniforms.member.member_1_;
            let _e146_ = select(1u, _e144_, (1u < _e144_));
            let _e148_ = select(_e144_, _e141_, (_e141_ < _e144_));
            let _e151_ = uniforms.member.member_4_;
            let _e153_ = select(_e151_, _e148_, (_e148_ < _e151_));
            let _e154_ = (_e128_ <= 1u);
            if _e154_ {
                phi_331_ = _e153_;
                phi_332_ = _e148_;
            } else {
                if (_e148_ > _e153_) {
                    phi_264_ = (_e148_ - _e153_);
                } else {
                    phi_264_ = 0u;
                }
                let _e158_ = phi_264_;
                if (_e158_ == 0u) {
                    phi_317_ = 0u;
                } else {
                    if _e154_ {
                        phi_316_ = _e158_;
                    } else {
                        let _e160_ = (_e158_ - 1u);
                        phi_275_ = 1u;
                        phi_278_ = _e128_;
                        loop {
                            let _e162_ = phi_275_;
                            let _e164_ = phi_278_;
                            local_25_ = _e162_;
                            local_26_ = _e164_;
                            let _e167_ = (_e164_ <= (_e160_ >> bitcast<u32>(1i)));
                            if _e167_ {
                                phi_276_ = (_e162_ << bitcast<u32>(1i));
                                phi_279_ = (_e164_ << bitcast<u32>(1i));
                            } else {
                                phi_276_ = u32();
                                phi_279_ = u32();
                            }
                            let _e173_ = phi_276_;
                            let _e175_ = phi_279_;
                            phi_275_ = _e173_;
                            phi_278_ = _e175_;
                            if !(_e167_) {
                                break;
                            }
                        }
                        phi_290_ = 0u;
                        let _e5993_ = local_25_;
                        phi_293_ = _e5993_;
                        phi_295_ = _e160_;
                        let _e5997_ = local_26_;
                        phi_297_ = _e5997_;
                        loop {
                            let _e178_ = phi_290_;
                            let _e180_ = phi_293_;
                            let _e182_ = phi_295_;
                            let _e184_ = phi_297_;
                            local_16_ = _e178_;
                            let _e185_ = (_e180_ > 0u);
                            if _e185_ {
                                if (_e182_ >= _e184_) {
                                    phi_311_ = (_e178_ | _e180_);
                                    phi_312_ = (_e182_ - _e184_);
                                } else {
                                    phi_311_ = _e178_;
                                    phi_312_ = _e182_;
                                }
                                let _e190_ = phi_311_;
                                let _e192_ = phi_312_;
                                phi_291_ = _e190_;
                                phi_294_ = (_e180_ >> bitcast<u32>(1i));
                                phi_296_ = _e192_;
                                phi_298_ = (_e184_ >> bitcast<u32>(1i));
                            } else {
                                phi_291_ = u32();
                                phi_294_ = u32();
                                phi_296_ = u32();
                                phi_298_ = u32();
                            }
                            let _e198_ = phi_291_;
                            let _e200_ = phi_294_;
                            let _e202_ = phi_296_;
                            let _e204_ = phi_298_;
                            phi_290_ = _e198_;
                            phi_293_ = _e200_;
                            phi_295_ = _e202_;
                            phi_297_ = _e204_;
                            if !(_e185_) {
                                break;
                            }
                        }
                        let _e207_ = local_16_;
                        phi_316_ = (_e207_ + 1u);
                    }
                    let _e210_ = phi_316_;
                    phi_317_ = _e210_;
                }
                let _e212_ = phi_317_;
                let _e214_ = (_e153_ + (_e109_.y * _e212_));
                let _e216_ = select(_e214_, _e148_, (_e148_ < _e214_));
                let _e217_ = (_e216_ + _e212_);
                phi_331_ = _e216_;
                phi_332_ = select(_e217_, _e148_, (_e148_ < _e217_));
            }
            let _e221_ = phi_331_;
            let _e223_ = phi_332_;
            let _e226_ = uniforms.member.member_7_;
            let _e228_ = (_e131_ * 25u);
            let _e229_ = (_e228_ < _e115_);
            if _e229_ {
                if _e229_ {
                } else {
                    break;
                }
                let _e232_ = quad_params.member[_e228_];
                phi_349_ = _e232_;
            } else {
                phi_349_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e234_ = phi_349_;
            let _e235_ = (_e228_ + 1u);
            let _e236_ = (_e235_ < _e115_);
            if _e236_ {
                if _e236_ {
                } else {
                    break;
                }
                let _e239_ = quad_params.member[_e235_];
                phi_362_ = _e239_;
            } else {
                phi_362_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e241_ = phi_362_;
            let _e242_ = (_e228_ + 2u);
            let _e243_ = (_e242_ < _e115_);
            if _e243_ {
                if _e243_ {
                } else {
                    break;
                }
                let _e246_ = quad_params.member[_e242_];
                phi_375_ = _e246_;
            } else {
                phi_375_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e248_ = phi_375_;
            let _e249_ = (_e228_ + 3u);
            let _e250_ = (_e249_ < _e115_);
            if _e250_ {
                if _e250_ {
                } else {
                    break;
                }
                let _e253_ = quad_params.member[_e249_];
                phi_388_ = _e253_;
            } else {
                phi_388_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e255_ = phi_388_;
            let _e256_ = (_e228_ + 4u);
            let _e257_ = (_e256_ < _e115_);
            if _e257_ {
                if _e257_ {
                } else {
                    break;
                }
                let _e260_ = quad_params.member[_e256_];
                phi_401_ = _e260_;
            } else {
                phi_401_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e262_ = phi_401_;
            let _e263_ = (_e228_ + 5u);
            let _e264_ = (_e263_ < _e115_);
            if _e264_ {
                if _e264_ {
                } else {
                    break;
                }
                let _e267_ = quad_params.member[_e263_];
                phi_414_ = _e267_;
            } else {
                phi_414_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e269_ = phi_414_;
            let _e270_ = (_e228_ + 6u);
            let _e271_ = (_e270_ < _e115_);
            if _e271_ {
                if _e271_ {
                } else {
                    break;
                }
                let _e274_ = quad_params.member[_e270_];
                phi_427_ = _e274_;
            } else {
                phi_427_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e276_ = phi_427_;
            let _e277_ = (_e228_ + 7u);
            let _e278_ = (_e277_ < _e115_);
            if _e278_ {
                if _e278_ {
                } else {
                    break;
                }
                let _e281_ = quad_params.member[_e277_];
                phi_440_ = _e281_;
            } else {
                phi_440_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e283_ = phi_440_;
            let _e284_ = (_e228_ + 8u);
            let _e285_ = (_e284_ < _e115_);
            if _e285_ {
                if _e285_ {
                } else {
                    break;
                }
                let _e288_ = quad_params.member[_e284_];
                phi_453_ = _e288_;
            } else {
                phi_453_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e290_ = phi_453_;
            let _e291_ = (_e228_ + 9u);
            let _e292_ = (_e291_ < _e115_);
            if _e292_ {
                if _e292_ {
                } else {
                    break;
                }
                let _e295_ = quad_params.member[_e291_];
                phi_466_ = _e295_;
            } else {
                phi_466_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e297_ = phi_466_;
            let _e298_ = (_e228_ + 10u);
            let _e299_ = (_e298_ < _e115_);
            if _e299_ {
                if _e299_ {
                } else {
                    break;
                }
                let _e302_ = quad_params.member[_e298_];
                phi_479_ = _e302_;
            } else {
                phi_479_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e304_ = phi_479_;
            let _e305_ = (_e228_ + 11u);
            let _e306_ = (_e305_ < _e115_);
            if _e306_ {
                if _e306_ {
                } else {
                    break;
                }
                let _e309_ = quad_params.member[_e305_];
                phi_492_ = _e309_;
            } else {
                phi_492_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e311_ = phi_492_;
            let _e312_ = (_e228_ + 12u);
            let _e313_ = (_e312_ < _e115_);
            if _e313_ {
                if _e313_ {
                } else {
                    break;
                }
                let _e316_ = quad_params.member[_e312_];
                phi_505_ = _e316_;
            } else {
                phi_505_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e318_ = phi_505_;
            let _e319_ = (_e228_ + 13u);
            let _e320_ = (_e319_ < _e115_);
            if _e320_ {
                if _e320_ {
                } else {
                    break;
                }
                let _e323_ = quad_params.member[_e319_];
                phi_518_ = _e323_;
            } else {
                phi_518_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e325_ = phi_518_;
            let _e326_ = (_e228_ + 14u);
            let _e327_ = (_e326_ < _e115_);
            if _e327_ {
                if _e327_ {
                } else {
                    break;
                }
                let _e330_ = quad_params.member[_e326_];
                phi_531_ = _e330_;
            } else {
                phi_531_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e332_ = phi_531_;
            let _e333_ = (_e228_ + 15u);
            let _e334_ = (_e333_ < _e115_);
            if _e334_ {
                if _e334_ {
                } else {
                    break;
                }
                let _e337_ = quad_params.member[_e333_];
                phi_544_ = _e337_;
            } else {
                phi_544_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e339_ = phi_544_;
            let _e340_ = (_e228_ + 16u);
            let _e341_ = (_e340_ < _e115_);
            if _e341_ {
                if _e341_ {
                } else {
                    break;
                }
                let _e344_ = quad_params.member[_e340_];
                phi_557_ = _e344_;
            } else {
                phi_557_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e346_ = phi_557_;
            let _e347_ = (_e228_ + 17u);
            let _e348_ = (_e347_ < _e115_);
            if _e348_ {
                if _e348_ {
                } else {
                    break;
                }
                let _e351_ = quad_params.member[_e347_];
                phi_570_ = _e351_;
            } else {
                phi_570_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e353_ = phi_570_;
            let _e354_ = (_e228_ + 18u);
            let _e355_ = (_e354_ < _e115_);
            if _e355_ {
                if _e355_ {
                } else {
                    break;
                }
                let _e358_ = quad_params.member[_e354_];
                phi_583_ = _e358_;
            } else {
                phi_583_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e360_ = phi_583_;
            let _e361_ = (_e228_ + 19u);
            let _e362_ = (_e361_ < _e115_);
            if _e362_ {
                if _e362_ {
                } else {
                    break;
                }
                let _e365_ = quad_params.member[_e361_];
                phi_596_ = _e365_;
            } else {
                phi_596_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e367_ = phi_596_;
            let _e368_ = (_e228_ + 20u);
            let _e369_ = (_e368_ < _e115_);
            if _e369_ {
                if _e369_ {
                } else {
                    break;
                }
                let _e372_ = quad_params.member[_e368_];
                phi_609_ = _e372_;
            } else {
                phi_609_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e374_ = phi_609_;
            let _e375_ = (_e228_ + 21u);
            let _e376_ = (_e375_ < _e115_);
            if _e376_ {
                if _e376_ {
                } else {
                    break;
                }
                let _e379_ = quad_params.member[_e375_];
                phi_622_ = _e379_;
            } else {
                phi_622_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e381_ = phi_622_;
            let _e382_ = (_e228_ + 22u);
            let _e383_ = (_e382_ < _e115_);
            if _e383_ {
                if _e383_ {
                } else {
                    break;
                }
                let _e386_ = quad_params.member[_e382_];
                phi_635_ = _e386_;
            } else {
                phi_635_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e388_ = phi_635_;
            let _e390_ = ((_e228_ + 23u) < _e115_);
            if _e390_ {
                if _e390_ {
                } else {
                    break;
                }
            }
            let _e391_ = (_e228_ + 24u);
            let _e392_ = (_e391_ < _e115_);
            if _e392_ {
                if _e392_ {
                } else {
                    break;
                }
                let _e395_ = quad_params.member[_e391_];
                phi_661_ = _e395_;
            } else {
                phi_661_ = vec4<f32>(0f, 0f, 0f, 0f);
            }
            let _e397_ = phi_661_;
            let _e399_ = bitcast<u32>(_e381_.x);
            let _e400_ = (_e399_ & 2139095040u);
            if (_e400_ != 2139095040u) {
                if (_e381_.x <= 0f) {
                    phi_695_ = 0f;
                } else {
                    if (_e381_.x >= 1f) {
                        phi_694_ = 1f;
                    } else {
                        if (_e400_ == 2139095040u) {
                            phi_688_ = true;
                        } else {
                            phi_688_ = (_e400_ == 0u);
                        }
                        let _e409_ = phi_688_;
                        phi_694_ = bitcast<f32>(select(_e399_, 0u, _e409_));
                    }
                    let _e413_ = phi_694_;
                    phi_695_ = _e413_;
                }
                let _e415_ = phi_695_;
                phi_702_ = _e415_;
            } else {
                phi_702_ = select(0f, 1f, (_e399_ == 2139095040u));
            }
            let _e417_ = phi_702_;
            let _e419_ = bitcast<u32>(_e381_.y);
            let _e420_ = (_e419_ & 2139095040u);
            if (_e420_ != 2139095040u) {
                if (_e381_.y <= 0f) {
                    phi_735_ = 0f;
                } else {
                    if (_e381_.y >= 1f) {
                        phi_734_ = 1f;
                    } else {
                        if (_e420_ == 2139095040u) {
                            phi_728_ = true;
                        } else {
                            phi_728_ = (_e420_ == 0u);
                        }
                        let _e429_ = phi_728_;
                        phi_734_ = bitcast<f32>(select(_e419_, 0u, _e429_));
                    }
                    let _e433_ = phi_734_;
                    phi_735_ = _e433_;
                }
                let _e435_ = phi_735_;
                phi_742_ = _e435_;
            } else {
                phi_742_ = select(0f, 1f, (_e419_ == 2139095040u));
            }
            let _e437_ = phi_742_;
            let _e439_ = bitcast<u32>(_e381_.z);
            let _e440_ = (_e439_ & 2139095040u);
            if (_e440_ != 2139095040u) {
                if (_e381_.z <= 0f) {
                    phi_775_ = 0f;
                } else {
                    if (_e381_.z >= 1f) {
                        phi_774_ = 1f;
                    } else {
                        if (_e440_ == 2139095040u) {
                            phi_768_ = true;
                        } else {
                            phi_768_ = (_e440_ == 0u);
                        }
                        let _e449_ = phi_768_;
                        phi_774_ = bitcast<f32>(select(_e439_, 0u, _e449_));
                    }
                    let _e453_ = phi_774_;
                    phi_775_ = _e453_;
                }
                let _e455_ = phi_775_;
                phi_782_ = _e455_;
            } else {
                phi_782_ = select(0f, 1f, (_e439_ == 2139095040u));
            }
            let _e457_ = phi_782_;
            let _e459_ = bitcast<u32>(_e381_.w);
            let _e460_ = (_e459_ & 2139095040u);
            if (_e460_ != 2139095040u) {
                if (_e381_.w <= 0f) {
                    phi_815_ = 0f;
                } else {
                    if (_e381_.w >= 1f) {
                        phi_814_ = 1f;
                    } else {
                        if (_e460_ == 2139095040u) {
                            phi_808_ = true;
                        } else {
                            phi_808_ = (_e460_ == 0u);
                        }
                        let _e469_ = phi_808_;
                        phi_814_ = bitcast<f32>(select(_e459_, 0u, _e469_));
                    }
                    let _e473_ = phi_814_;
                    phi_815_ = _e473_;
                }
                let _e475_ = phi_815_;
                phi_822_ = _e475_;
            } else {
                phi_822_ = select(0f, 1f, (_e459_ == 2139095040u));
            }
            let _e477_ = phi_822_;
            if _e154_ {
                let _e482_ = (_e153_ > 0u);
                if _e482_ {
                    let _e483_ = (_e153_ < _e146_);
                    if _e483_ {
                        let _e486_ = (((_e131_ * _e146_) + _e153_) * 8u);
                        let _e487_ = (_e486_ < _e117_);
                        if _e487_ {
                            if _e487_ {
                            } else {
                                break;
                            }
                            let _e490_ = tile_checkpoints.member[_e486_];
                            phi_854_ = _e490_;
                        } else {
                            phi_854_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e492_ = phi_854_;
                        let _e494_ = bitcast<u32>(_e492_.x);
                        let _e495_ = (_e494_ & 2139095040u);
                        if (_e495_ == 2139095040u) {
                            phi_866_ = true;
                        } else {
                            phi_866_ = (_e495_ == 0u);
                        }
                        let _e499_ = phi_866_;
                        let _e503_ = bitcast<u32>(_e492_.y);
                        let _e504_ = (_e503_ & 2139095040u);
                        if (_e504_ == 2139095040u) {
                            phi_883_ = true;
                        } else {
                            phi_883_ = (_e504_ == 0u);
                        }
                        let _e508_ = phi_883_;
                        let _e512_ = bitcast<u32>(_e492_.z);
                        let _e513_ = (_e512_ & 2139095040u);
                        if (_e513_ == 2139095040u) {
                            phi_900_ = true;
                        } else {
                            phi_900_ = (_e513_ == 0u);
                        }
                        let _e517_ = phi_900_;
                        let _e521_ = bitcast<u32>(_e492_.w);
                        let _e522_ = (_e521_ & 2139095040u);
                        if (_e522_ == 2139095040u) {
                            phi_917_ = true;
                        } else {
                            phi_917_ = (_e522_ == 0u);
                        }
                        let _e526_ = phi_917_;
                        let _e530_ = (_e486_ + 1u);
                        let _e531_ = (_e530_ < _e117_);
                        if _e531_ {
                            if _e531_ {
                            } else {
                                break;
                            }
                            let _e534_ = tile_checkpoints.member[_e530_];
                            phi_936_ = _e534_;
                        } else {
                            phi_936_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e536_ = phi_936_;
                        let _e538_ = bitcast<u32>(_e536_.x);
                        let _e539_ = (_e538_ & 2139095040u);
                        if (_e539_ == 2139095040u) {
                            phi_948_ = true;
                        } else {
                            phi_948_ = (_e539_ == 0u);
                        }
                        let _e543_ = phi_948_;
                        let _e547_ = bitcast<u32>(_e536_.y);
                        let _e548_ = (_e547_ & 2139095040u);
                        if (_e548_ == 2139095040u) {
                            phi_965_ = true;
                        } else {
                            phi_965_ = (_e548_ == 0u);
                        }
                        let _e552_ = phi_965_;
                        let _e556_ = bitcast<u32>(_e536_.z);
                        let _e557_ = (_e556_ & 2139095040u);
                        if (_e557_ == 2139095040u) {
                            phi_982_ = true;
                        } else {
                            phi_982_ = (_e557_ == 0u);
                        }
                        let _e561_ = phi_982_;
                        let _e565_ = bitcast<u32>(_e536_.w);
                        let _e566_ = (_e565_ & 2139095040u);
                        if (_e566_ == 2139095040u) {
                            phi_999_ = true;
                        } else {
                            phi_999_ = (_e566_ == 0u);
                        }
                        let _e570_ = phi_999_;
                        let _e574_ = (_e486_ + 2u);
                        let _e575_ = (_e574_ < _e117_);
                        if _e575_ {
                            if _e575_ {
                            } else {
                                break;
                            }
                            let _e578_ = tile_checkpoints.member[_e574_];
                            phi_1018_ = _e578_;
                        } else {
                            phi_1018_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e580_ = phi_1018_;
                        let _e582_ = bitcast<u32>(_e580_.x);
                        let _e583_ = (_e582_ & 2139095040u);
                        if (_e583_ == 2139095040u) {
                            phi_1030_ = true;
                        } else {
                            phi_1030_ = (_e583_ == 0u);
                        }
                        let _e587_ = phi_1030_;
                        let _e591_ = bitcast<u32>(_e580_.y);
                        let _e592_ = (_e591_ & 2139095040u);
                        if (_e592_ == 2139095040u) {
                            phi_1047_ = true;
                        } else {
                            phi_1047_ = (_e592_ == 0u);
                        }
                        let _e596_ = phi_1047_;
                        let _e600_ = bitcast<u32>(_e580_.z);
                        let _e601_ = (_e600_ & 2139095040u);
                        if (_e601_ == 2139095040u) {
                            phi_1064_ = true;
                        } else {
                            phi_1064_ = (_e601_ == 0u);
                        }
                        let _e605_ = phi_1064_;
                        let _e609_ = bitcast<u32>(_e580_.w);
                        let _e610_ = (_e609_ & 2139095040u);
                        if (_e610_ == 2139095040u) {
                            phi_1081_ = true;
                        } else {
                            phi_1081_ = (_e610_ == 0u);
                        }
                        let _e614_ = phi_1081_;
                        let _e618_ = (_e486_ + 3u);
                        let _e619_ = (_e618_ < _e117_);
                        if _e619_ {
                            if _e619_ {
                            } else {
                                break;
                            }
                            let _e622_ = tile_checkpoints.member[_e618_];
                            phi_1100_ = _e622_;
                        } else {
                            phi_1100_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e624_ = phi_1100_;
                        let _e626_ = bitcast<u32>(_e624_.x);
                        let _e627_ = (_e626_ & 2139095040u);
                        if (_e627_ == 2139095040u) {
                            phi_1112_ = true;
                        } else {
                            phi_1112_ = (_e627_ == 0u);
                        }
                        let _e631_ = phi_1112_;
                        let _e635_ = bitcast<u32>(_e624_.y);
                        let _e636_ = (_e635_ & 2139095040u);
                        if (_e636_ == 2139095040u) {
                            phi_1129_ = true;
                        } else {
                            phi_1129_ = (_e636_ == 0u);
                        }
                        let _e640_ = phi_1129_;
                        let _e644_ = bitcast<u32>(_e624_.z);
                        let _e645_ = (_e644_ & 2139095040u);
                        if (_e645_ == 2139095040u) {
                            phi_1146_ = true;
                        } else {
                            phi_1146_ = (_e645_ == 0u);
                        }
                        let _e649_ = phi_1146_;
                        let _e653_ = bitcast<u32>(_e624_.w);
                        let _e654_ = (_e653_ & 2139095040u);
                        if (_e654_ == 2139095040u) {
                            phi_1163_ = true;
                        } else {
                            phi_1163_ = (_e654_ == 0u);
                        }
                        let _e658_ = phi_1163_;
                        let _e662_ = (_e486_ + 4u);
                        let _e663_ = (_e662_ < _e117_);
                        if _e663_ {
                            if _e663_ {
                            } else {
                                break;
                            }
                            let _e666_ = tile_checkpoints.member[_e662_];
                            phi_1182_ = _e666_;
                        } else {
                            phi_1182_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e668_ = phi_1182_;
                        let _e670_ = bitcast<u32>(_e668_.x);
                        let _e671_ = (_e670_ & 2139095040u);
                        if (_e671_ == 2139095040u) {
                            phi_1194_ = true;
                        } else {
                            phi_1194_ = (_e671_ == 0u);
                        }
                        let _e675_ = phi_1194_;
                        let _e679_ = bitcast<u32>(_e668_.y);
                        let _e680_ = (_e679_ & 2139095040u);
                        if (_e680_ == 2139095040u) {
                            phi_1211_ = true;
                        } else {
                            phi_1211_ = (_e680_ == 0u);
                        }
                        let _e684_ = phi_1211_;
                        let _e688_ = bitcast<u32>(_e668_.z);
                        let _e689_ = (_e688_ & 2139095040u);
                        if (_e689_ == 2139095040u) {
                            phi_1228_ = true;
                        } else {
                            phi_1228_ = (_e689_ == 0u);
                        }
                        let _e693_ = phi_1228_;
                        let _e697_ = bitcast<u32>(_e668_.w);
                        let _e698_ = (_e697_ & 2139095040u);
                        if (_e698_ == 2139095040u) {
                            phi_1245_ = true;
                        } else {
                            phi_1245_ = (_e698_ == 0u);
                        }
                        let _e702_ = phi_1245_;
                        let _e706_ = (_e486_ + 5u);
                        let _e707_ = (_e706_ < _e117_);
                        if _e707_ {
                            if _e707_ {
                            } else {
                                break;
                            }
                            let _e710_ = tile_checkpoints.member[_e706_];
                            phi_1264_ = _e710_;
                        } else {
                            phi_1264_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e712_ = phi_1264_;
                        let _e714_ = bitcast<u32>(_e712_.x);
                        let _e715_ = (_e714_ & 2139095040u);
                        if (_e715_ == 2139095040u) {
                            phi_1276_ = true;
                        } else {
                            phi_1276_ = (_e715_ == 0u);
                        }
                        let _e719_ = phi_1276_;
                        let _e723_ = bitcast<u32>(_e712_.y);
                        let _e724_ = (_e723_ & 2139095040u);
                        if (_e724_ == 2139095040u) {
                            phi_1293_ = true;
                        } else {
                            phi_1293_ = (_e724_ == 0u);
                        }
                        let _e728_ = phi_1293_;
                        let _e732_ = bitcast<u32>(_e712_.z);
                        let _e733_ = (_e732_ & 2139095040u);
                        if (_e733_ == 2139095040u) {
                            phi_1310_ = true;
                        } else {
                            phi_1310_ = (_e733_ == 0u);
                        }
                        let _e737_ = phi_1310_;
                        let _e741_ = bitcast<u32>(_e712_.w);
                        let _e742_ = (_e741_ & 2139095040u);
                        if (_e742_ == 2139095040u) {
                            phi_1327_ = true;
                        } else {
                            phi_1327_ = (_e742_ == 0u);
                        }
                        let _e746_ = phi_1327_;
                        let _e750_ = (_e486_ + 6u);
                        let _e751_ = (_e750_ < _e117_);
                        if _e751_ {
                            if _e751_ {
                            } else {
                                break;
                            }
                            let _e754_ = tile_checkpoints.member[_e750_];
                            phi_1346_ = _e754_;
                        } else {
                            phi_1346_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e756_ = phi_1346_;
                        let _e758_ = bitcast<u32>(_e756_.x);
                        let _e759_ = (_e758_ & 2139095040u);
                        if (_e759_ == 2139095040u) {
                            phi_1358_ = true;
                        } else {
                            phi_1358_ = (_e759_ == 0u);
                        }
                        let _e763_ = phi_1358_;
                        let _e765_ = bitcast<f32>(select(_e758_, 0u, _e763_));
                        if (_e765_ < -1000000000000000000f) {
                            phi_1375_ = -1000000000000000000f;
                        } else {
                            phi_1375_ = select(_e765_, 1000000000000000000f, (_e765_ > 1000000000000000000f));
                        }
                        let _e770_ = phi_1375_;
                        let _e772_ = bitcast<u32>(_e756_.y);
                        let _e773_ = (_e772_ & 2139095040u);
                        if (_e773_ == 2139095040u) {
                            phi_1387_ = true;
                        } else {
                            phi_1387_ = (_e773_ == 0u);
                        }
                        let _e777_ = phi_1387_;
                        let _e779_ = bitcast<f32>(select(_e772_, 0u, _e777_));
                        if (_e779_ < -1000000000000000000f) {
                            phi_1404_ = -1000000000000000000f;
                        } else {
                            phi_1404_ = select(_e779_, 1000000000000000000f, (_e779_ > 1000000000000000000f));
                        }
                        let _e784_ = phi_1404_;
                        let _e786_ = bitcast<u32>(_e756_.z);
                        let _e787_ = (_e786_ & 2139095040u);
                        if (_e787_ == 2139095040u) {
                            phi_1416_ = true;
                        } else {
                            phi_1416_ = (_e787_ == 0u);
                        }
                        let _e791_ = phi_1416_;
                        let _e793_ = bitcast<f32>(select(_e786_, 0u, _e791_));
                        if (_e793_ < -1000000000000000000f) {
                            phi_1433_ = -1000000000000000000f;
                        } else {
                            phi_1433_ = select(_e793_, 1000000000000000000f, (_e793_ > 1000000000000000000f));
                        }
                        let _e798_ = phi_1433_;
                        let _e800_ = bitcast<u32>(_e756_.w);
                        let _e801_ = (_e800_ & 2139095040u);
                        if (_e801_ == 2139095040u) {
                            phi_1445_ = true;
                        } else {
                            phi_1445_ = (_e801_ == 0u);
                        }
                        let _e805_ = phi_1445_;
                        let _e807_ = bitcast<f32>(select(_e800_, 0u, _e805_));
                        if (_e807_ < -1000000000000000000f) {
                            phi_1462_ = -1000000000000000000f;
                        } else {
                            phi_1462_ = select(_e807_, 1000000000000000000f, (_e807_ > 1000000000000000000f));
                        }
                        let _e812_ = phi_1462_;
                        let _e814_ = (_e486_ + 7u);
                        let _e815_ = (_e814_ < _e117_);
                        if _e815_ {
                            if _e815_ {
                            } else {
                                break;
                            }
                            let _e818_ = tile_checkpoints.member[_e814_];
                            phi_1476_ = _e818_;
                        } else {
                            phi_1476_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e820_ = phi_1476_;
                        let _e822_ = bitcast<u32>(_e820_.x);
                        let _e823_ = (_e822_ & 2139095040u);
                        if (_e823_ == 2139095040u) {
                            phi_1488_ = true;
                        } else {
                            phi_1488_ = (_e823_ == 0u);
                        }
                        let _e827_ = phi_1488_;
                        let _e829_ = bitcast<f32>(select(_e822_, 0u, _e827_));
                        if (_e829_ < -1000000000000000000f) {
                            phi_1505_ = -1000000000000000000f;
                        } else {
                            phi_1505_ = select(_e829_, 1000000000000000000f, (_e829_ > 1000000000000000000f));
                        }
                        let _e834_ = phi_1505_;
                        let _e836_ = bitcast<u32>(_e820_.y);
                        let _e837_ = (_e836_ & 2139095040u);
                        if (_e837_ == 2139095040u) {
                            phi_1517_ = true;
                        } else {
                            phi_1517_ = (_e837_ == 0u);
                        }
                        let _e841_ = phi_1517_;
                        let _e843_ = bitcast<f32>(select(_e836_, 0u, _e841_));
                        if (_e843_ < -1000000000000000000f) {
                            phi_1534_ = -1000000000000000000f;
                        } else {
                            phi_1534_ = select(_e843_, 1000000000000000000f, (_e843_ > 1000000000000000000f));
                        }
                        let _e848_ = phi_1534_;
                        let _e850_ = bitcast<u32>(_e820_.z);
                        let _e851_ = (_e850_ & 2139095040u);
                        if (_e851_ == 2139095040u) {
                            phi_1546_ = true;
                        } else {
                            phi_1546_ = (_e851_ == 0u);
                        }
                        let _e855_ = phi_1546_;
                        let _e857_ = bitcast<f32>(select(_e850_, 0u, _e855_));
                        if (_e857_ < -1000000000000000000f) {
                            phi_1563_ = -1000000000000000000f;
                        } else {
                            phi_1563_ = select(_e857_, 1000000000000000000f, (_e857_ > 1000000000000000000f));
                        }
                        let _e862_ = phi_1563_;
                        let _e864_ = bitcast<u32>(_e820_.w);
                        let _e865_ = (_e864_ & 2139095040u);
                        if (_e865_ == 2139095040u) {
                            phi_1575_ = true;
                        } else {
                            phi_1575_ = (_e865_ == 0u);
                        }
                        let _e869_ = phi_1575_;
                        let _e871_ = bitcast<f32>(select(_e864_, 0u, _e869_));
                        if (_e871_ < -1000000000000000000f) {
                            phi_1592_ = -1000000000000000000f;
                        } else {
                            phi_1592_ = select(_e871_, 1000000000000000000f, (_e871_ > 1000000000000000000f));
                        }
                        let _e876_ = phi_1592_;
                        phi_1603_ = type_32_(type_27_(type_24_(array<f32, 4>(bitcast<f32>(select(_e494_, 0u, _e499_)), bitcast<f32>(select(_e503_, 0u, _e508_)), bitcast<f32>(select(_e512_, 0u, _e517_)), bitcast<f32>(select(_e521_, 0u, _e526_)))), type_24_(array<f32, 4>(bitcast<f32>(select(_e538_, 0u, _e543_)), bitcast<f32>(select(_e547_, 0u, _e552_)), bitcast<f32>(select(_e556_, 0u, _e561_)), bitcast<f32>(select(_e565_, 0u, _e570_))))), type_27_(type_24_(array<f32, 4>(bitcast<f32>(select(_e582_, 0u, _e587_)), bitcast<f32>(select(_e591_, 0u, _e596_)), bitcast<f32>(select(_e600_, 0u, _e605_)), bitcast<f32>(select(_e609_, 0u, _e614_)))), type_24_(array<f32, 4>(bitcast<f32>(select(_e626_, 0u, _e631_)), bitcast<f32>(select(_e635_, 0u, _e640_)), bitcast<f32>(select(_e644_, 0u, _e649_)), bitcast<f32>(select(_e653_, 0u, _e658_))))), type_27_(type_24_(array<f32, 4>(bitcast<f32>(select(_e670_, 0u, _e675_)), bitcast<f32>(select(_e679_, 0u, _e684_)), bitcast<f32>(select(_e688_, 0u, _e693_)), bitcast<f32>(select(_e697_, 0u, _e702_)))), type_24_(array<f32, 4>(bitcast<f32>(select(_e714_, 0u, _e719_)), bitcast<f32>(select(_e723_, 0u, _e728_)), bitcast<f32>(select(_e732_, 0u, _e737_)), bitcast<f32>(select(_e741_, 0u, _e746_))))), type_27_(type_24_(array<f32, 4>(_e770_, _e784_, _e798_, _e812_)), type_24_(array<f32, 4>(_e834_, _e848_, _e862_, _e876_))));
                    } else {
                        phi_1603_ = type_32_();
                    }
                    let _e892_ = phi_1603_;
                    phi_1605_ = _e892_;
                    phi_1606_ = _e483_;
                    phi_1609_ = select(true, false, _e483_);
                } else {
                    phi_1605_ = type_32_();
                    phi_1606_ = bool();
                    phi_1609_ = false;
                }
                let _e895_ = phi_1605_;
                let _e897_ = phi_1606_;
                let _e899_ = phi_1609_;
                phi_1610_ = _e895_;
                phi_1611_ = _e897_;
                phi_1613_ = select(true, false, _e482_);
                phi_1614_ = _e899_;
            } else {
                phi_1610_ = type_32_();
                phi_1611_ = bool();
                phi_1613_ = true;
                phi_1614_ = false;
            }
            let _e902_ = phi_1610_;
            let _e904_ = phi_1611_;
            let _e906_ = phi_1613_;
            let _e908_ = phi_1614_;
            let _e910_ = select(_e908_, true, _e906_);
            if _e910_ {
                phi_1636_ = type_32_(type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))), type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))), type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))), type_27_(type_24_(array<f32, 4>(0f, 0f, 0f, 0f)), type_24_(array<f32, 4>(0f, 0f, 0f, 0f))));
            } else {
                phi_1636_ = _e902_;
            }
            let _e912_ = phi_1636_;
            phi_17856_ = false;
            phi_1640_ = type_16_(_e221_, _e223_);
            phi_1643_ = _e912_;
            loop {
                let _e916_ = phi_17856_;
                let _e918_ = phi_1640_;
                let _e920_ = phi_1643_;
                if (_e918_.member < _e918_.member_1_) {
                    phi_1641_ = type_16_((_e918_.member + 1u), _e918_.member_1_);
                    phi_1666_ = type_16_(1u, _e918_.member);
                } else {
                    phi_1641_ = _e918_;
                    phi_1666_ = type_16_(0u, u32());
                }
                let _e933_ = phi_1641_;
                let _e935_ = phi_1666_;
                let _e939_ = (bitcast<i32>(_e935_.member) != 0i);
                if _e939_ {
                    let _e940_ = (_e935_.member_1_ * 256u);
                    if (_e134_ > _e940_) {
                        phi_1680_ = (_e134_ - _e940_);
                    } else {
                        phi_1680_ = 0u;
                    }
                    let _e944_ = phi_1680_;
                    let _e946_ = select(_e944_, 256u, (256u < _e944_));
                    let _e948_ = (_e120_ < _e946_);
                    let _e949_ = (_e120_ == 0u);
                    if _e949_ {
                        if select(_e904_, select(_e904_, false, _e906_), _e910_) {
                            phi_1700_ = select(true, false, (_e935_.member_1_ == _e153_));
                        } else {
                            phi_1700_ = true;
                        }
                        let _e953_ = phi_1700_;
                        if _e953_ {
                            let _e956_ = (((_e131_ * _e146_) + _e935_.member_1_) * 8u);
                            let _e981_ = bitcast<u32>(_e920_.member.member.member[0]);
                            let _e982_ = (_e981_ & 2139095040u);
                            if (_e982_ == 2139095040u) {
                                phi_1749_ = true;
                            } else {
                                phi_1749_ = (_e982_ == 0u);
                            }
                            let _e986_ = phi_1749_;
                            let _e989_ = bitcast<u32>(_e920_.member.member.member[1]);
                            let _e990_ = (_e989_ & 2139095040u);
                            if (_e990_ == 2139095040u) {
                                phi_1765_ = true;
                            } else {
                                phi_1765_ = (_e990_ == 0u);
                            }
                            let _e994_ = phi_1765_;
                            let _e997_ = bitcast<u32>(_e920_.member.member.member[2]);
                            let _e998_ = (_e997_ & 2139095040u);
                            if (_e998_ == 2139095040u) {
                                phi_1781_ = true;
                            } else {
                                phi_1781_ = (_e998_ == 0u);
                            }
                            let _e1002_ = phi_1781_;
                            let _e1005_ = bitcast<u32>(_e920_.member.member.member[3]);
                            let _e1006_ = (_e1005_ & 2139095040u);
                            if (_e1006_ == 2139095040u) {
                                phi_1797_ = true;
                            } else {
                                phi_1797_ = (_e1006_ == 0u);
                            }
                            let _e1010_ = phi_1797_;
                            let _e1014_ = (_e956_ < _e117_);
                            if _e1014_ {
                                if _e1014_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e956_] = vec4<f32>(bitcast<f32>(select(_e981_, 0u, _e986_)), bitcast<f32>(select(_e989_, 0u, _e994_)), bitcast<f32>(select(_e997_, 0u, _e1002_)), bitcast<f32>(select(_e1005_, 0u, _e1010_)));
                            }
                            let _e1017_ = (_e956_ + 1u);
                            let _e1022_ = bitcast<u32>(_e920_.member.member_1_.member[0]);
                            let _e1023_ = (_e1022_ & 2139095040u);
                            if (_e1023_ == 2139095040u) {
                                phi_1829_ = true;
                            } else {
                                phi_1829_ = (_e1023_ == 0u);
                            }
                            let _e1027_ = phi_1829_;
                            let _e1030_ = bitcast<u32>(_e920_.member.member_1_.member[1]);
                            let _e1031_ = (_e1030_ & 2139095040u);
                            if (_e1031_ == 2139095040u) {
                                phi_1845_ = true;
                            } else {
                                phi_1845_ = (_e1031_ == 0u);
                            }
                            let _e1035_ = phi_1845_;
                            let _e1038_ = bitcast<u32>(_e920_.member.member_1_.member[2]);
                            let _e1039_ = (_e1038_ & 2139095040u);
                            if (_e1039_ == 2139095040u) {
                                phi_1861_ = true;
                            } else {
                                phi_1861_ = (_e1039_ == 0u);
                            }
                            let _e1043_ = phi_1861_;
                            let _e1046_ = bitcast<u32>(_e920_.member.member_1_.member[3]);
                            let _e1047_ = (_e1046_ & 2139095040u);
                            if (_e1047_ == 2139095040u) {
                                phi_1877_ = true;
                            } else {
                                phi_1877_ = (_e1047_ == 0u);
                            }
                            let _e1051_ = phi_1877_;
                            let _e1055_ = (_e1017_ < _e117_);
                            if _e1055_ {
                                if _e1055_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1017_] = vec4<f32>(bitcast<f32>(select(_e1022_, 0u, _e1027_)), bitcast<f32>(select(_e1030_, 0u, _e1035_)), bitcast<f32>(select(_e1038_, 0u, _e1043_)), bitcast<f32>(select(_e1046_, 0u, _e1051_)));
                            }
                            let _e1058_ = (_e956_ + 2u);
                            let _e1063_ = bitcast<u32>(_e920_.member_1_.member.member[0]);
                            let _e1064_ = (_e1063_ & 2139095040u);
                            if (_e1064_ == 2139095040u) {
                                phi_1909_ = true;
                            } else {
                                phi_1909_ = (_e1064_ == 0u);
                            }
                            let _e1068_ = phi_1909_;
                            let _e1071_ = bitcast<u32>(_e920_.member_1_.member.member[1]);
                            let _e1072_ = (_e1071_ & 2139095040u);
                            if (_e1072_ == 2139095040u) {
                                phi_1925_ = true;
                            } else {
                                phi_1925_ = (_e1072_ == 0u);
                            }
                            let _e1076_ = phi_1925_;
                            let _e1079_ = bitcast<u32>(_e920_.member_1_.member.member[2]);
                            let _e1080_ = (_e1079_ & 2139095040u);
                            if (_e1080_ == 2139095040u) {
                                phi_1941_ = true;
                            } else {
                                phi_1941_ = (_e1080_ == 0u);
                            }
                            let _e1084_ = phi_1941_;
                            let _e1087_ = bitcast<u32>(_e920_.member_1_.member.member[3]);
                            let _e1088_ = (_e1087_ & 2139095040u);
                            if (_e1088_ == 2139095040u) {
                                phi_1957_ = true;
                            } else {
                                phi_1957_ = (_e1088_ == 0u);
                            }
                            let _e1092_ = phi_1957_;
                            let _e1096_ = (_e1058_ < _e117_);
                            if _e1096_ {
                                if _e1096_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1058_] = vec4<f32>(bitcast<f32>(select(_e1063_, 0u, _e1068_)), bitcast<f32>(select(_e1071_, 0u, _e1076_)), bitcast<f32>(select(_e1079_, 0u, _e1084_)), bitcast<f32>(select(_e1087_, 0u, _e1092_)));
                            }
                            let _e1099_ = (_e956_ + 3u);
                            let _e1104_ = bitcast<u32>(_e920_.member_1_.member_1_.member[0]);
                            let _e1105_ = (_e1104_ & 2139095040u);
                            if (_e1105_ == 2139095040u) {
                                phi_1989_ = true;
                            } else {
                                phi_1989_ = (_e1105_ == 0u);
                            }
                            let _e1109_ = phi_1989_;
                            let _e1112_ = bitcast<u32>(_e920_.member_1_.member_1_.member[1]);
                            let _e1113_ = (_e1112_ & 2139095040u);
                            if (_e1113_ == 2139095040u) {
                                phi_2005_ = true;
                            } else {
                                phi_2005_ = (_e1113_ == 0u);
                            }
                            let _e1117_ = phi_2005_;
                            let _e1120_ = bitcast<u32>(_e920_.member_1_.member_1_.member[2]);
                            let _e1121_ = (_e1120_ & 2139095040u);
                            if (_e1121_ == 2139095040u) {
                                phi_2021_ = true;
                            } else {
                                phi_2021_ = (_e1121_ == 0u);
                            }
                            let _e1125_ = phi_2021_;
                            let _e1128_ = bitcast<u32>(_e920_.member_1_.member_1_.member[3]);
                            let _e1129_ = (_e1128_ & 2139095040u);
                            if (_e1129_ == 2139095040u) {
                                phi_2037_ = true;
                            } else {
                                phi_2037_ = (_e1129_ == 0u);
                            }
                            let _e1133_ = phi_2037_;
                            let _e1137_ = (_e1099_ < _e117_);
                            if _e1137_ {
                                if _e1137_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1099_] = vec4<f32>(bitcast<f32>(select(_e1104_, 0u, _e1109_)), bitcast<f32>(select(_e1112_, 0u, _e1117_)), bitcast<f32>(select(_e1120_, 0u, _e1125_)), bitcast<f32>(select(_e1128_, 0u, _e1133_)));
                            }
                            let _e1140_ = (_e956_ + 4u);
                            let _e1145_ = bitcast<u32>(_e920_.member_2_.member.member[0]);
                            let _e1146_ = (_e1145_ & 2139095040u);
                            if (_e1146_ == 2139095040u) {
                                phi_2069_ = true;
                            } else {
                                phi_2069_ = (_e1146_ == 0u);
                            }
                            let _e1150_ = phi_2069_;
                            let _e1153_ = bitcast<u32>(_e920_.member_2_.member.member[1]);
                            let _e1154_ = (_e1153_ & 2139095040u);
                            if (_e1154_ == 2139095040u) {
                                phi_2085_ = true;
                            } else {
                                phi_2085_ = (_e1154_ == 0u);
                            }
                            let _e1158_ = phi_2085_;
                            let _e1161_ = bitcast<u32>(_e920_.member_2_.member.member[2]);
                            let _e1162_ = (_e1161_ & 2139095040u);
                            if (_e1162_ == 2139095040u) {
                                phi_2101_ = true;
                            } else {
                                phi_2101_ = (_e1162_ == 0u);
                            }
                            let _e1166_ = phi_2101_;
                            let _e1169_ = bitcast<u32>(_e920_.member_2_.member.member[3]);
                            let _e1170_ = (_e1169_ & 2139095040u);
                            if (_e1170_ == 2139095040u) {
                                phi_2117_ = true;
                            } else {
                                phi_2117_ = (_e1170_ == 0u);
                            }
                            let _e1174_ = phi_2117_;
                            let _e1178_ = (_e1140_ < _e117_);
                            if _e1178_ {
                                if _e1178_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1140_] = vec4<f32>(bitcast<f32>(select(_e1145_, 0u, _e1150_)), bitcast<f32>(select(_e1153_, 0u, _e1158_)), bitcast<f32>(select(_e1161_, 0u, _e1166_)), bitcast<f32>(select(_e1169_, 0u, _e1174_)));
                            }
                            let _e1181_ = (_e956_ + 5u);
                            let _e1186_ = bitcast<u32>(_e920_.member_2_.member_1_.member[0]);
                            let _e1187_ = (_e1186_ & 2139095040u);
                            if (_e1187_ == 2139095040u) {
                                phi_2149_ = true;
                            } else {
                                phi_2149_ = (_e1187_ == 0u);
                            }
                            let _e1191_ = phi_2149_;
                            let _e1194_ = bitcast<u32>(_e920_.member_2_.member_1_.member[1]);
                            let _e1195_ = (_e1194_ & 2139095040u);
                            if (_e1195_ == 2139095040u) {
                                phi_2165_ = true;
                            } else {
                                phi_2165_ = (_e1195_ == 0u);
                            }
                            let _e1199_ = phi_2165_;
                            let _e1202_ = bitcast<u32>(_e920_.member_2_.member_1_.member[2]);
                            let _e1203_ = (_e1202_ & 2139095040u);
                            if (_e1203_ == 2139095040u) {
                                phi_2181_ = true;
                            } else {
                                phi_2181_ = (_e1203_ == 0u);
                            }
                            let _e1207_ = phi_2181_;
                            let _e1210_ = bitcast<u32>(_e920_.member_2_.member_1_.member[3]);
                            let _e1211_ = (_e1210_ & 2139095040u);
                            if (_e1211_ == 2139095040u) {
                                phi_2197_ = true;
                            } else {
                                phi_2197_ = (_e1211_ == 0u);
                            }
                            let _e1215_ = phi_2197_;
                            let _e1219_ = (_e1181_ < _e117_);
                            if _e1219_ {
                                if _e1219_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1181_] = vec4<f32>(bitcast<f32>(select(_e1186_, 0u, _e1191_)), bitcast<f32>(select(_e1194_, 0u, _e1199_)), bitcast<f32>(select(_e1202_, 0u, _e1207_)), bitcast<f32>(select(_e1210_, 0u, _e1215_)));
                            }
                            let _e1222_ = (_e956_ + 6u);
                            let _e1227_ = bitcast<u32>(_e920_.member_3_.member.member[0]);
                            let _e1228_ = (_e1227_ & 2139095040u);
                            if (_e1228_ == 2139095040u) {
                                phi_2229_ = true;
                            } else {
                                phi_2229_ = (_e1228_ == 0u);
                            }
                            let _e1232_ = phi_2229_;
                            let _e1235_ = bitcast<u32>(_e920_.member_3_.member.member[1]);
                            let _e1236_ = (_e1235_ & 2139095040u);
                            if (_e1236_ == 2139095040u) {
                                phi_2245_ = true;
                            } else {
                                phi_2245_ = (_e1236_ == 0u);
                            }
                            let _e1240_ = phi_2245_;
                            let _e1243_ = bitcast<u32>(_e920_.member_3_.member.member[2]);
                            let _e1244_ = (_e1243_ & 2139095040u);
                            if (_e1244_ == 2139095040u) {
                                phi_2261_ = true;
                            } else {
                                phi_2261_ = (_e1244_ == 0u);
                            }
                            let _e1248_ = phi_2261_;
                            let _e1251_ = bitcast<u32>(_e920_.member_3_.member.member[3]);
                            let _e1252_ = (_e1251_ & 2139095040u);
                            if (_e1252_ == 2139095040u) {
                                phi_2277_ = true;
                            } else {
                                phi_2277_ = (_e1252_ == 0u);
                            }
                            let _e1256_ = phi_2277_;
                            let _e1260_ = (_e1222_ < _e117_);
                            if _e1260_ {
                                if _e1260_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1222_] = vec4<f32>(bitcast<f32>(select(_e1227_, 0u, _e1232_)), bitcast<f32>(select(_e1235_, 0u, _e1240_)), bitcast<f32>(select(_e1243_, 0u, _e1248_)), bitcast<f32>(select(_e1251_, 0u, _e1256_)));
                            }
                            let _e1263_ = (_e956_ + 7u);
                            let _e1268_ = bitcast<u32>(_e920_.member_3_.member_1_.member[0]);
                            let _e1269_ = (_e1268_ & 2139095040u);
                            if (_e1269_ == 2139095040u) {
                                phi_2309_ = true;
                            } else {
                                phi_2309_ = (_e1269_ == 0u);
                            }
                            let _e1273_ = phi_2309_;
                            let _e1276_ = bitcast<u32>(_e920_.member_3_.member_1_.member[1]);
                            let _e1277_ = (_e1276_ & 2139095040u);
                            if (_e1277_ == 2139095040u) {
                                phi_2325_ = true;
                            } else {
                                phi_2325_ = (_e1277_ == 0u);
                            }
                            let _e1281_ = phi_2325_;
                            let _e1284_ = bitcast<u32>(_e920_.member_3_.member_1_.member[2]);
                            let _e1285_ = (_e1284_ & 2139095040u);
                            if (_e1285_ == 2139095040u) {
                                phi_2341_ = true;
                            } else {
                                phi_2341_ = (_e1285_ == 0u);
                            }
                            let _e1289_ = phi_2341_;
                            let _e1292_ = bitcast<u32>(_e920_.member_3_.member_1_.member[3]);
                            let _e1293_ = (_e1292_ & 2139095040u);
                            if (_e1293_ == 2139095040u) {
                                phi_2357_ = true;
                            } else {
                                phi_2357_ = (_e1293_ == 0u);
                            }
                            let _e1297_ = phi_2357_;
                            let _e1301_ = (_e1263_ < _e117_);
                            if _e1301_ {
                                if _e1301_ {
                                } else {
                                    phi_18515_ = true;
                                    break;
                                }
                                tile_checkpoints.member[_e1263_] = vec4<f32>(bitcast<f32>(select(_e1268_, 0u, _e1273_)), bitcast<f32>(select(_e1276_, 0u, _e1281_)), bitcast<f32>(select(_e1284_, 0u, _e1289_)), bitcast<f32>(select(_e1292_, 0u, _e1297_)));
                            }
                        }
                    }
                    let _e1304_ = (_e120_ < 512u);
                    if _e1304_ {
                    } else {
                        phi_18515_ = true;
                        break;
                    }
                    lane_buf[_e120_] = vec4<f32>(0f, 0f, 0f, 0f);
                    let _e1306_ = (256u + _e120_);
                    let _e1307_ = (_e1306_ < 512u);
                    if _e1307_ {
                    } else {
                        phi_18515_ = true;
                        break;
                    }
                    lane_buf[_e1306_] = vec4<f32>(0f, 0f, 0f, 0f);
                    let _e1310_ = ((_e131_ * _e134_) + (_e940_ + _e120_));
                    let _e1311_ = (_e1310_ * 2u);
                    if _e948_ {
                        let _e1312_ = (_e1311_ < _e113_);
                        if _e1312_ {
                            if _e1312_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            let _e1315_ = core_inputs.member[_e1311_];
                            phi_2404_ = _e1315_;
                        } else {
                            phi_2404_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e1317_ = phi_2404_;
                        let _e1318_ = (_e1311_ + 1u);
                        let _e1319_ = (_e1318_ < _e113_);
                        if _e1319_ {
                            if _e1319_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            let _e1322_ = core_inputs.member[_e1318_];
                            phi_2417_ = _e1322_;
                        } else {
                            phi_2417_ = vec4<f32>(0f, 0f, 0f, 0f);
                        }
                        let _e1324_ = phi_2417_;
                        phi_2418_ = _e1324_;
                        phi_2419_ = _e1317_;
                    } else {
                        phi_2418_ = vec4<f32>(0f, 0f, 0f, 0f);
                        phi_2419_ = vec4<f32>(0f, 0f, 0f, 0f);
                    }
                    let _e1326_ = phi_2418_;
                    let _e1328_ = phi_2419_;
                    local = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_1_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_2_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_3_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_4_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    local_5_ = array<f32, 4>(0f, 0f, 0f, 0f);
                    phi_17850_ = _e916_;
                    phi_2436_ = type_16_(0u, 4u);
                    phi_2439_ = _e920_;
                    loop {
                        let _e1340_ = phi_17850_;
                        let _e1342_ = phi_2436_;
                        let _e1344_ = phi_2439_;
                        local_23_ = _e1344_;
                        local_24_ = _e1344_;
                        if (_e1342_.member < _e1342_.member_1_) {
                            phi_2437_ = type_16_((_e1342_.member + 1u), _e1342_.member_1_);
                            phi_2462_ = type_16_(1u, _e1342_.member);
                        } else {
                            phi_2437_ = _e1342_;
                            phi_2462_ = type_16_(0u, u32());
                        }
                        let _e1357_ = phi_2437_;
                        let _e1359_ = phi_2462_;
                        let _e1363_ = (bitcast<i32>(_e1359_.member) != 0i);
                        if _e1363_ {
                            let _e1364_ = (_e1359_.member_1_ & 3u);
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2480_ = _e304_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2480_ = _e304_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2480_ = _e304_.z;
                                    break;
                                }
                                default: {
                                    phi_2480_ = _e304_.w;
                                    break;
                                }
                            }
                            let _e1371_ = phi_2480_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2491_ = _e311_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2491_ = _e311_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2491_ = _e311_.z;
                                    break;
                                }
                                default: {
                                    phi_2491_ = _e311_.w;
                                    break;
                                }
                            }
                            let _e1378_ = phi_2491_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2502_ = _e318_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2502_ = _e318_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2502_ = _e318_.z;
                                    break;
                                }
                                default: {
                                    phi_2502_ = _e318_.w;
                                    break;
                                }
                            }
                            let _e1385_ = phi_2502_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2513_ = _e325_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2513_ = _e325_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2513_ = _e325_.z;
                                    break;
                                }
                                default: {
                                    phi_2513_ = _e325_.w;
                                    break;
                                }
                            }
                            let _e1392_ = phi_2513_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2524_ = _e332_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2524_ = _e332_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2524_ = _e332_.z;
                                    break;
                                }
                                default: {
                                    phi_2524_ = _e332_.w;
                                    break;
                                }
                            }
                            let _e1399_ = phi_2524_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2535_ = _e339_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2535_ = _e339_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2535_ = _e339_.z;
                                    break;
                                }
                                default: {
                                    phi_2535_ = _e339_.w;
                                    break;
                                }
                            }
                            let _e1406_ = phi_2535_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2557_ = _e234_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2557_ = _e234_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2557_ = _e234_.z;
                                    break;
                                }
                                default: {
                                    phi_2557_ = _e234_.w;
                                    break;
                                }
                            }
                            let _e1413_ = phi_2557_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2568_ = _e241_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2568_ = _e241_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2568_ = _e241_.z;
                                    break;
                                }
                                default: {
                                    phi_2568_ = _e241_.w;
                                    break;
                                }
                            }
                            let _e1420_ = phi_2568_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2579_ = _e248_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2579_ = _e248_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2579_ = _e248_.z;
                                    break;
                                }
                                default: {
                                    phi_2579_ = _e248_.w;
                                    break;
                                }
                            }
                            let _e1427_ = phi_2579_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2590_ = _e255_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2590_ = _e255_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2590_ = _e255_.z;
                                    break;
                                }
                                default: {
                                    phi_2590_ = _e255_.w;
                                    break;
                                }
                            }
                            let _e1434_ = phi_2590_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2601_ = _e262_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2601_ = _e262_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2601_ = _e262_.z;
                                    break;
                                }
                                default: {
                                    phi_2601_ = _e262_.w;
                                    break;
                                }
                            }
                            let _e1441_ = phi_2601_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2612_ = _e269_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2612_ = _e269_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2612_ = _e269_.z;
                                    break;
                                }
                                default: {
                                    phi_2612_ = _e269_.w;
                                    break;
                                }
                            }
                            let _e1448_ = phi_2612_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2623_ = _e276_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2623_ = _e276_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2623_ = _e276_.z;
                                    break;
                                }
                                default: {
                                    phi_2623_ = _e276_.w;
                                    break;
                                }
                            }
                            let _e1455_ = phi_2623_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2634_ = _e283_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2634_ = _e283_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2634_ = _e283_.z;
                                    break;
                                }
                                default: {
                                    phi_2634_ = _e283_.w;
                                    break;
                                }
                            }
                            let _e1462_ = phi_2634_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2645_ = _e290_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2645_ = _e290_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2645_ = _e290_.z;
                                    break;
                                }
                                default: {
                                    phi_2645_ = _e290_.w;
                                    break;
                                }
                            }
                            let _e1469_ = phi_2645_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2656_ = _e297_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2656_ = _e297_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2656_ = _e297_.z;
                                    break;
                                }
                                default: {
                                    phi_2656_ = _e297_.w;
                                    break;
                                }
                            }
                            let _e1476_ = phi_2656_;
                            let _e1477_ = bitcast<u32>(_e1371_);
                            let _e1478_ = (_e1477_ & 2139095040u);
                            if (_e1478_ != 2139095040u) {
                                if (_e1371_ <= 0f) {
                                    phi_2688_ = 0f;
                                } else {
                                    if (_e1371_ >= 0.999995f) {
                                        phi_2687_ = 0.999995f;
                                    } else {
                                        if (_e1478_ == 2139095040u) {
                                            phi_2681_ = true;
                                        } else {
                                            phi_2681_ = (_e1478_ == 0u);
                                        }
                                        let _e1487_ = phi_2681_;
                                        phi_2687_ = bitcast<f32>(select(_e1477_, 0u, _e1487_));
                                    }
                                    let _e1491_ = phi_2687_;
                                    phi_2688_ = _e1491_;
                                }
                                let _e1493_ = phi_2688_;
                                phi_2695_ = _e1493_;
                            } else {
                                phi_2695_ = select(0f, 0.999995f, (_e1477_ == 2139095040u));
                            }
                            let _e1495_ = phi_2695_;
                            let _e1496_ = bitcast<u32>(_e1378_);
                            let _e1497_ = (_e1496_ & 2139095040u);
                            if (_e1497_ != 2139095040u) {
                                if (_e1378_ <= 0f) {
                                    phi_2727_ = 0f;
                                } else {
                                    if (_e1378_ >= 1f) {
                                        phi_2726_ = 1f;
                                    } else {
                                        if (_e1497_ == 2139095040u) {
                                            phi_2720_ = true;
                                        } else {
                                            phi_2720_ = (_e1497_ == 0u);
                                        }
                                        let _e1506_ = phi_2720_;
                                        phi_2726_ = bitcast<f32>(select(_e1496_, 0u, _e1506_));
                                    }
                                    let _e1510_ = phi_2726_;
                                    phi_2727_ = _e1510_;
                                }
                                let _e1512_ = phi_2727_;
                                phi_2734_ = _e1512_;
                            } else {
                                phi_2734_ = select(0f, 1f, (_e1496_ == 2139095040u));
                            }
                            let _e1514_ = phi_2734_;
                            let _e1515_ = bitcast<u32>(_e1385_);
                            let _e1516_ = (_e1515_ & 2139095040u);
                            if (_e1516_ != 2139095040u) {
                                if (_e1385_ <= 0f) {
                                    phi_2766_ = 0f;
                                } else {
                                    if (_e1385_ >= 1f) {
                                        phi_2765_ = 1f;
                                    } else {
                                        if (_e1516_ == 2139095040u) {
                                            phi_2759_ = true;
                                        } else {
                                            phi_2759_ = (_e1516_ == 0u);
                                        }
                                        let _e1525_ = phi_2759_;
                                        phi_2765_ = bitcast<f32>(select(_e1515_, 0u, _e1525_));
                                    }
                                    let _e1529_ = phi_2765_;
                                    phi_2766_ = _e1529_;
                                }
                                let _e1531_ = phi_2766_;
                                phi_2773_ = _e1531_;
                            } else {
                                phi_2773_ = select(0f, 1f, (_e1515_ == 2139095040u));
                            }
                            let _e1533_ = phi_2773_;
                            let _e1534_ = bitcast<u32>(_e1392_);
                            let _e1535_ = (_e1534_ & 2139095040u);
                            if (_e1535_ != 2139095040u) {
                                if (_e1392_ <= 0f) {
                                    phi_2805_ = 0f;
                                } else {
                                    if (_e1392_ >= 64f) {
                                        phi_2804_ = 64f;
                                    } else {
                                        if (_e1535_ == 2139095040u) {
                                            phi_2798_ = true;
                                        } else {
                                            phi_2798_ = (_e1535_ == 0u);
                                        }
                                        let _e1544_ = phi_2798_;
                                        phi_2804_ = bitcast<f32>(select(_e1534_, 0u, _e1544_));
                                    }
                                    let _e1548_ = phi_2804_;
                                    phi_2805_ = _e1548_;
                                }
                                let _e1550_ = phi_2805_;
                                phi_2812_ = _e1550_;
                            } else {
                                phi_2812_ = select(0f, 64f, (_e1534_ == 2139095040u));
                            }
                            let _e1552_ = phi_2812_;
                            let _e1553_ = bitcast<u32>(_e1399_);
                            let _e1554_ = (_e1553_ & 2139095040u);
                            if (_e1554_ != 2139095040u) {
                                if (_e1399_ <= 0f) {
                                    phi_2844_ = 0f;
                                } else {
                                    if (_e1399_ >= 1f) {
                                        phi_2843_ = 1f;
                                    } else {
                                        if (_e1554_ == 2139095040u) {
                                            phi_2837_ = true;
                                        } else {
                                            phi_2837_ = (_e1554_ == 0u);
                                        }
                                        let _e1563_ = phi_2837_;
                                        phi_2843_ = bitcast<f32>(select(_e1553_, 0u, _e1563_));
                                    }
                                    let _e1567_ = phi_2843_;
                                    phi_2844_ = _e1567_;
                                }
                                let _e1569_ = phi_2844_;
                                phi_2851_ = _e1569_;
                            } else {
                                phi_2851_ = select(0f, 1f, (_e1553_ == 2139095040u));
                            }
                            let _e1571_ = phi_2851_;
                            let _e1572_ = bitcast<u32>(_e1406_);
                            let _e1573_ = (_e1572_ & 2139095040u);
                            if (_e1573_ != 2139095040u) {
                                if (_e1406_ <= 0f) {
                                    phi_2883_ = 0f;
                                } else {
                                    if (_e1406_ >= 1f) {
                                        phi_2882_ = 1f;
                                    } else {
                                        if (_e1573_ == 2139095040u) {
                                            phi_2876_ = true;
                                        } else {
                                            phi_2876_ = (_e1573_ == 0u);
                                        }
                                        let _e1582_ = phi_2876_;
                                        phi_2882_ = bitcast<f32>(select(_e1572_, 0u, _e1582_));
                                    }
                                    let _e1586_ = phi_2882_;
                                    phi_2883_ = _e1586_;
                                }
                                let _e1588_ = phi_2883_;
                                phi_2890_ = _e1588_;
                            } else {
                                phi_2890_ = select(0f, 1f, (_e1572_ == 2139095040u));
                            }
                            let _e1590_ = phi_2890_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2961_ = _e346_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2961_ = _e346_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2961_ = _e346_.z;
                                    break;
                                }
                                default: {
                                    phi_2961_ = _e346_.w;
                                    break;
                                }
                            }
                            let _e1597_ = phi_2961_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2972_ = _e353_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2972_ = _e353_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2972_ = _e353_.z;
                                    break;
                                }
                                default: {
                                    phi_2972_ = _e353_.w;
                                    break;
                                }
                            }
                            let _e1604_ = phi_2972_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2983_ = _e360_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2983_ = _e360_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2983_ = _e360_.z;
                                    break;
                                }
                                default: {
                                    phi_2983_ = _e360_.w;
                                    break;
                                }
                            }
                            let _e1611_ = phi_2983_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_2994_ = _e367_.x;
                                    break;
                                }
                                case 1: {
                                    phi_2994_ = _e367_.y;
                                    break;
                                }
                                case 2: {
                                    phi_2994_ = _e367_.z;
                                    break;
                                }
                                default: {
                                    phi_2994_ = _e367_.w;
                                    break;
                                }
                            }
                            let _e1618_ = phi_2994_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_3005_ = _e374_.x;
                                    break;
                                }
                                case 1: {
                                    phi_3005_ = _e374_.y;
                                    break;
                                }
                                case 2: {
                                    phi_3005_ = _e374_.z;
                                    break;
                                }
                                default: {
                                    phi_3005_ = _e374_.w;
                                    break;
                                }
                            }
                            let _e1625_ = phi_3005_;
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    phi_3056_ = _e397_.x;
                                    break;
                                }
                                case 1: {
                                    phi_3056_ = _e397_.y;
                                    break;
                                }
                                case 2: {
                                    phi_3056_ = _e397_.z;
                                    break;
                                }
                                default: {
                                    phi_3056_ = _e397_.w;
                                    break;
                                }
                            }
                            let _e1632_ = phi_3056_;
                            if _e948_ {
                                switch bitcast<i32>(_e1364_) {
                                    case 0: {
                                        phi_3070_ = _e1328_.x;
                                        break;
                                    }
                                    case 1: {
                                        phi_3070_ = _e1328_.y;
                                        break;
                                    }
                                    case 2: {
                                        phi_3070_ = _e1328_.z;
                                        break;
                                    }
                                    default: {
                                        phi_3070_ = _e1328_.w;
                                        break;
                                    }
                                }
                                let _e1639_ = phi_3070_;
                                let _e1640_ = bitcast<u32>(_e1639_);
                                let _e1641_ = (_e1640_ & 2139095040u);
                                if (_e1641_ == 2139095040u) {
                                    phi_3081_ = true;
                                } else {
                                    phi_3081_ = (_e1641_ == 0u);
                                }
                                let _e1645_ = phi_3081_;
                                phi_3087_ = bitcast<f32>(select(_e1640_, 0u, _e1645_));
                            } else {
                                phi_3087_ = 0f;
                            }
                            let _e1649_ = phi_3087_;
                            if _e948_ {
                                switch bitcast<i32>(_e1364_) {
                                    case 0: {
                                        phi_3101_ = _e1326_.x;
                                        break;
                                    }
                                    case 1: {
                                        phi_3101_ = _e1326_.y;
                                        break;
                                    }
                                    case 2: {
                                        phi_3101_ = _e1326_.z;
                                        break;
                                    }
                                    default: {
                                        phi_3101_ = _e1326_.w;
                                        break;
                                    }
                                }
                                let _e1656_ = phi_3101_;
                                let _e1657_ = bitcast<u32>(_e1656_);
                                let _e1658_ = (_e1657_ & 2139095040u);
                                if (_e1658_ == 2139095040u) {
                                    phi_3112_ = true;
                                } else {
                                    phi_3112_ = (_e1658_ == 0u);
                                }
                                let _e1662_ = phi_3112_;
                                phi_3118_ = bitcast<f32>(select(_e1657_, 0u, _e1662_));
                            } else {
                                phi_3118_ = 0f;
                            }
                            let _e1666_ = phi_3118_;
                            if (_e226_ == 0u) {
                                let _e2145_ = (1024u + _e120_);
                                let _e2146_ = (_e2145_ < 1536u);
                                if _e2146_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2145_] = _e1649_;
                                let _e2148_ = (1280u + _e120_);
                                let _e2149_ = (_e2148_ < 1536u);
                                if _e2149_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2148_] = _e1666_;
                                workgroupBarrier();
                                if _e949_ {
                                    local_6_ = array<f32, 4>(_e920_.member.member.member[0], _e920_.member.member.member[1], _e920_.member.member.member[2], _e920_.member.member.member[3]);
                                    let _e2156_ = (_e1364_ < 4u);
                                    if _e2156_ {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2158_ = local_6_[_e1364_];
                                    local_7_ = array<f32, 4>(_e920_.member_1_.member.member[0], _e920_.member_1_.member.member[1], _e920_.member_1_.member.member[2], _e920_.member_1_.member.member[3]);
                                    if _e2156_ {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2165_ = local_7_[_e1364_];
                                    phi_3168_ = type_16_(0u, _e946_);
                                    phi_3171_ = _e2165_;
                                    phi_3173_ = _e2158_;
                                    loop {
                                        let _e2168_ = phi_3168_;
                                        let _e2170_ = phi_3171_;
                                        let _e2172_ = phi_3173_;
                                        local_17_ = _e2172_;
                                        local_18_ = _e2170_;
                                        if (_e2168_.member < _e2168_.member_1_) {
                                            phi_3169_ = type_16_((_e2168_.member + 1u), _e2168_.member_1_);
                                            phi_3196_ = type_16_(1u, _e2168_.member);
                                        } else {
                                            phi_3169_ = _e2168_;
                                            phi_3196_ = type_16_(0u, u32());
                                        }
                                        let _e2185_ = phi_3169_;
                                        let _e2187_ = phi_3196_;
                                        let _e2191_ = (bitcast<i32>(_e2187_.member) != 0i);
                                        if _e2191_ {
                                            let _e2192_ = (_e2187_.member_1_ & 255u);
                                            let _e2193_ = (1024u + _e2192_);
                                            if (_e2193_ < 1536u) {
                                            } else {
                                                phi_18133_ = true;
                                                break;
                                            }
                                            let _e2196_ = scan_scratch[_e2193_];
                                            let _e2197_ = (_e1597_ * 0.5f);
                                            if (_e2197_ != _e2197_) {
                                                phi_12029_ = true;
                                            } else {
                                                phi_12029_ = (0.00001f >= _e2197_);
                                            }
                                            let _e2201_ = phi_12029_;
                                            let _e2202_ = select(_e2197_, 0.00001f, _e2201_);
                                            if (_e1434_ != _e1434_) {
                                                phi_12089_ = true;
                                            } else {
                                                phi_12089_ = (0f >= _e1434_);
                                            }
                                            let _e2206_ = phi_12089_;
                                            if (_e1632_ != _e1632_) {
                                                phi_12149_ = true;
                                            } else {
                                                phi_12149_ = (0f >= _e1632_);
                                            }
                                            let _e2211_ = phi_12149_;
                                            let _e2213_ = (_e2172_ * _e2172_);
                                            if (_e2213_ != _e2213_) {
                                                phi_12209_ = true;
                                            } else {
                                                phi_12209_ = (64f <= _e2213_);
                                            }
                                            let _e2217_ = phi_12209_;
                                            let _e2220_ = (select(_e1434_, 0f, _e2206_) + (select(_e1632_, 0f, _e2211_) * select(_e2213_, 64f, _e2217_)));
                                            let _e2223_ = (1f + (_e2202_ * (_e2202_ + _e2220_)));
                                            let _e2226_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2223_)));
                                            let _e2230_ = (_e2226_ + (_e2226_ * (1f - (_e2223_ * _e2226_))));
                                            let _e2234_ = (_e2230_ + (_e2230_ * (1f - (_e2223_ * _e2230_))));
                                            let _e2235_ = (_e2202_ * _e2234_);
                                            let _e2238_ = (_e2196_ - _e2170_);
                                            let _e2241_ = ((_e2234_ * _e2172_) + (_e2235_ * _e2238_));
                                            let _e2245_ = ((_e2170_ + (_e2235_ * _e2172_)) + ((_e2202_ * _e2235_) * _e2238_));
                                            let _e2248_ = bitcast<u32>(((_e2241_ + _e2241_) - _e2172_));
                                            let _e2249_ = (_e2248_ & 2139095040u);
                                            if (_e2249_ == 2139095040u) {
                                                phi_3364_ = 0u;
                                            } else {
                                                if (_e2249_ == 0u) {
                                                    phi_3363_ = (_e2248_ & 2147483648u);
                                                } else {
                                                    phi_3363_ = _e2248_;
                                                }
                                                let _e2254_ = phi_3363_;
                                                phi_3364_ = _e2254_;
                                            }
                                            let _e2256_ = phi_3364_;
                                            let _e2260_ = bitcast<u32>(((_e2245_ + _e2245_) - _e2170_));
                                            let _e2261_ = (_e2260_ & 2139095040u);
                                            if (_e2261_ == 2139095040u) {
                                                phi_3416_ = 0u;
                                            } else {
                                                if (_e2261_ == 0u) {
                                                    phi_3415_ = (_e2260_ & 2147483648u);
                                                } else {
                                                    phi_3415_ = _e2260_;
                                                }
                                                let _e2266_ = phi_3415_;
                                                phi_3416_ = _e2266_;
                                            }
                                            let _e2268_ = phi_3416_;
                                            if (_e2192_ < 1536u) {
                                            } else {
                                                phi_18133_ = true;
                                                break;
                                            }
                                            scan_scratch[_e2192_] = (((((_e2245_ * _e1448_) + (_e2241_ * _e1455_)) + (((_e2196_ - (_e2220_ * _e2241_)) - _e2245_) * _e1462_)) + ((_e2196_ - ((_e2220_ + _e2220_) * _e2241_)) * _e1469_)) + (_e2196_ * _e1476_));
                                            let _e2286_ = (256u + _e2192_);
                                            if (_e2286_ < 1536u) {
                                            } else {
                                                phi_18133_ = true;
                                                break;
                                            }
                                            scan_scratch[_e2286_] = _e2241_;
                                            phi_3172_ = bitcast<f32>(_e2268_);
                                            phi_3174_ = bitcast<f32>(_e2256_);
                                        } else {
                                            phi_3172_ = f32();
                                            phi_3174_ = f32();
                                        }
                                        let _e2290_ = phi_3172_;
                                        let _e2292_ = phi_3174_;
                                        phi_3168_ = _e2185_;
                                        phi_3171_ = _e2290_;
                                        phi_3173_ = _e2292_;
                                        phi_18133_ = _e1340_;
                                        if !(_e2191_) {
                                            break;
                                        }
                                    }
                                    let _e2295_ = phi_18133_;
                                    phi_18498_ = _e2295_;
                                    if _e2295_ {
                                        break;
                                    }
                                    let _e2298_ = local_17_;
                                    aux_scratch[0] = _e2298_;
                                    let _e2301_ = local_18_;
                                    aux_scratch[1] = _e2301_;
                                    phi_18177_ = _e2295_;
                                } else {
                                    phi_18177_ = _e1340_;
                                }
                                let _e2303_ = phi_18177_;
                                if (_e120_ == 1u) {
                                    local_8_ = array<f32, 4>(_e920_.member.member_1_.member[0], _e920_.member.member_1_.member[1], _e920_.member.member_1_.member[2], _e920_.member.member_1_.member[3]);
                                    let _e2310_ = (_e1364_ < 4u);
                                    if _e2310_ {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2312_ = local_8_[_e1364_];
                                    local_9_ = array<f32, 4>(_e920_.member_1_.member_1_.member[0], _e920_.member_1_.member_1_.member[1], _e920_.member_1_.member_1_.member[2], _e920_.member_1_.member_1_.member[3]);
                                    if _e2310_ {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2319_ = local_9_[_e1364_];
                                    phi_3538_ = type_16_(0u, _e946_);
                                    phi_3541_ = _e2319_;
                                    phi_3543_ = _e2312_;
                                    loop {
                                        let _e2322_ = phi_3538_;
                                        let _e2324_ = phi_3541_;
                                        let _e2326_ = phi_3543_;
                                        local_19_ = _e2326_;
                                        local_20_ = _e2324_;
                                        if (_e2322_.member < _e2322_.member_1_) {
                                            phi_3539_ = type_16_((_e2322_.member + 1u), _e2322_.member_1_);
                                            phi_3566_ = type_16_(1u, _e2322_.member);
                                        } else {
                                            phi_3539_ = _e2322_;
                                            phi_3566_ = type_16_(0u, u32());
                                        }
                                        let _e2339_ = phi_3539_;
                                        let _e2341_ = phi_3566_;
                                        let _e2345_ = (bitcast<i32>(_e2341_.member) != 0i);
                                        if _e2345_ {
                                            let _e2346_ = (_e2341_.member_1_ & 255u);
                                            let _e2347_ = (1280u + _e2346_);
                                            if (_e2347_ < 1536u) {
                                            } else {
                                                phi_18173_ = true;
                                                break;
                                            }
                                            let _e2350_ = scan_scratch[_e2347_];
                                            let _e2351_ = (_e1597_ * 0.5f);
                                            if (_e2351_ != _e2351_) {
                                                phi_12269_ = true;
                                            } else {
                                                phi_12269_ = (0.00001f >= _e2351_);
                                            }
                                            let _e2355_ = phi_12269_;
                                            let _e2356_ = select(_e2351_, 0.00001f, _e2355_);
                                            if (_e1434_ != _e1434_) {
                                                phi_12329_ = true;
                                            } else {
                                                phi_12329_ = (0f >= _e1434_);
                                            }
                                            let _e2360_ = phi_12329_;
                                            if (_e1632_ != _e1632_) {
                                                phi_12389_ = true;
                                            } else {
                                                phi_12389_ = (0f >= _e1632_);
                                            }
                                            let _e2365_ = phi_12389_;
                                            let _e2367_ = (_e2326_ * _e2326_);
                                            if (_e2367_ != _e2367_) {
                                                phi_12449_ = true;
                                            } else {
                                                phi_12449_ = (64f <= _e2367_);
                                            }
                                            let _e2371_ = phi_12449_;
                                            let _e2374_ = (select(_e1434_, 0f, _e2360_) + (select(_e1632_, 0f, _e2365_) * select(_e2367_, 64f, _e2371_)));
                                            let _e2377_ = (1f + (_e2356_ * (_e2356_ + _e2374_)));
                                            let _e2380_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2377_)));
                                            let _e2384_ = (_e2380_ + (_e2380_ * (1f - (_e2377_ * _e2380_))));
                                            let _e2388_ = (_e2384_ + (_e2384_ * (1f - (_e2377_ * _e2384_))));
                                            let _e2389_ = (_e2356_ * _e2388_);
                                            let _e2392_ = (_e2350_ - _e2324_);
                                            let _e2395_ = ((_e2388_ * _e2326_) + (_e2389_ * _e2392_));
                                            let _e2399_ = ((_e2324_ + (_e2389_ * _e2326_)) + ((_e2356_ * _e2389_) * _e2392_));
                                            let _e2402_ = bitcast<u32>(((_e2395_ + _e2395_) - _e2326_));
                                            let _e2403_ = (_e2402_ & 2139095040u);
                                            if (_e2403_ == 2139095040u) {
                                                phi_3734_ = 0u;
                                            } else {
                                                if (_e2403_ == 0u) {
                                                    phi_3733_ = (_e2402_ & 2147483648u);
                                                } else {
                                                    phi_3733_ = _e2402_;
                                                }
                                                let _e2408_ = phi_3733_;
                                                phi_3734_ = _e2408_;
                                            }
                                            let _e2410_ = phi_3734_;
                                            let _e2414_ = bitcast<u32>(((_e2399_ + _e2399_) - _e2324_));
                                            let _e2415_ = (_e2414_ & 2139095040u);
                                            if (_e2415_ == 2139095040u) {
                                                phi_3786_ = 0u;
                                            } else {
                                                if (_e2415_ == 0u) {
                                                    phi_3785_ = (_e2414_ & 2147483648u);
                                                } else {
                                                    phi_3785_ = _e2414_;
                                                }
                                                let _e2420_ = phi_3785_;
                                                phi_3786_ = _e2420_;
                                            }
                                            let _e2422_ = phi_3786_;
                                            let _e2438_ = (512u + _e2346_);
                                            if (_e2438_ < 1536u) {
                                            } else {
                                                phi_18173_ = true;
                                                break;
                                            }
                                            scan_scratch[_e2438_] = (((((_e2399_ * _e1448_) + (_e2395_ * _e1455_)) + (((_e2350_ - (_e2374_ * _e2395_)) - _e2399_) * _e1462_)) + ((_e2350_ - ((_e2374_ + _e2374_) * _e2395_)) * _e1469_)) + (_e2350_ * _e1476_));
                                            let _e2441_ = (768u + _e2346_);
                                            if (_e2441_ < 1536u) {
                                            } else {
                                                phi_18173_ = true;
                                                break;
                                            }
                                            scan_scratch[_e2441_] = _e2395_;
                                            phi_3542_ = bitcast<f32>(_e2422_);
                                            phi_3544_ = bitcast<f32>(_e2410_);
                                        } else {
                                            phi_3542_ = f32();
                                            phi_3544_ = f32();
                                        }
                                        let _e2445_ = phi_3542_;
                                        let _e2447_ = phi_3544_;
                                        phi_3538_ = _e2339_;
                                        phi_3541_ = _e2445_;
                                        phi_3543_ = _e2447_;
                                        phi_18173_ = _e2303_;
                                        if !(_e2345_) {
                                            break;
                                        }
                                    }
                                    let _e2450_ = phi_18173_;
                                    phi_18498_ = _e2450_;
                                    if _e2450_ {
                                        break;
                                    }
                                    let _e2453_ = local_19_;
                                    aux_scratch[2] = _e2453_;
                                    let _e2456_ = local_20_;
                                    aux_scratch[3] = _e2456_;
                                    phi_18225_ = _e2450_;
                                } else {
                                    phi_18225_ = _e2303_;
                                }
                                let _e2458_ = phi_18225_;
                                workgroupBarrier();
                                let _e2460_ = aux_scratch[0];
                                let _e2461_ = (_e1364_ < 4u);
                                if _e2461_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_2_[_e1364_] = _e2460_;
                                let _e2464_ = aux_scratch[1];
                                if _e2461_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_4_[_e1364_] = _e2464_;
                                let _e2467_ = aux_scratch[2];
                                if _e2461_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_3_[_e1364_] = _e2467_;
                                let _e2470_ = aux_scratch[3];
                                if _e2461_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_5_[_e1364_] = _e2470_;
                                if _e948_ {
                                    if (_e120_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2484_ = scan_scratch[_e120_];
                                    phi_3950_ = _e2484_;
                                } else {
                                    if (_e120_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    scan_scratch[_e120_] = 0f;
                                    if (_e1306_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    scan_scratch[_e1306_] = 0f;
                                    let _e2476_ = (512u + _e120_);
                                    if (_e2476_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    scan_scratch[_e2476_] = 0f;
                                    let _e2479_ = (768u + _e120_);
                                    if (_e2479_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    scan_scratch[_e2479_] = 0f;
                                    phi_3950_ = 0f;
                                }
                                let _e2486_ = phi_3950_;
                                if _e948_ {
                                    let _e2487_ = (512u + _e120_);
                                    if (_e2487_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2490_ = scan_scratch[_e2487_];
                                    phi_3962_ = _e2490_;
                                } else {
                                    phi_3962_ = 0f;
                                }
                                let _e2492_ = phi_3962_;
                                if _e2146_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2145_] = 0f;
                                if _e2149_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2148_] = 0f;
                                workgroupBarrier();
                                phi_18218_ = _e2458_;
                                phi_5271_ = _e2486_;
                                phi_5272_ = _e2492_;
                            } else {
                                if _e948_ {
                                    let _e1667_ = (_e1420_ + _e1420_);
                                    let _e1668_ = (_e1427_ + _e1427_);
                                    phi_4000_ = type_26_(((_e1413_ + _e1413_) - 1f), -(_e1667_), _e1667_, (1f - _e1668_), (_e1667_ * _e1649_), (_e1668_ * _e1649_));
                                } else {
                                    phi_4000_ = type_26_(1f, 0f, 0f, 1f, 0f, 0f);
                                }
                                let _e1677_ = phi_4000_;
                                local_10_ = array<f32, 4>(_e920_.member.member.member[0], _e920_.member.member.member[1], _e920_.member.member.member[2], _e920_.member.member.member[3]);
                                let _e1683_ = (_e1364_ < 4u);
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e1685_ = local_10_[_e1364_];
                                local_11_ = array<f32, 4>(_e920_.member_1_.member.member[0], _e920_.member_1_.member.member[1], _e920_.member_1_.member.member[2], _e920_.member_1_.member.member[3]);
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e1692_ = local_11_[_e1364_];
                                let _e1693_ = array<f32, 2>(_e1685_, _e1692_);
                                let _e1694_ = (_e120_ & 255u);
                                let _e1695_ = (_e1694_ & 255u);
                                let _e1697_ = (_e1695_ < 1536u);
                                if _e1697_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1695_] = _e1677_.member;
                                let _e1700_ = (256u + _e1695_);
                                let _e1701_ = (_e1700_ < 1536u);
                                if _e1701_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1700_] = _e1677_.member_1_;
                                let _e1704_ = (512u + _e1695_);
                                let _e1705_ = (_e1704_ < 1536u);
                                if _e1705_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1704_] = _e1677_.member_2_;
                                let _e1708_ = (768u + _e1695_);
                                let _e1709_ = (_e1708_ < 1536u);
                                if _e1709_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1708_] = _e1677_.member_3_;
                                let _e1712_ = (1024u + _e1695_);
                                let _e1713_ = (_e1712_ < 1536u);
                                if _e1713_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1712_] = _e1677_.member_4_;
                                let _e1716_ = (1280u + _e1695_);
                                let _e1717_ = (_e1716_ < 1536u);
                                if _e1717_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1716_] = _e1677_.member_5_;
                                workgroupBarrier();
                                phi_4078_ = 1u;
                                phi_4081_ = _e1677_;
                                loop {
                                    let _e1720_ = phi_4078_;
                                    let _e1722_ = phi_4081_;
                                    let _e1723_ = (_e1720_ < 256u);
                                    if _e1723_ {
                                        if (_e1694_ >= _e1720_) {
                                            let _e1726_ = ((_e1694_ - _e1720_) & 255u);
                                            if (_e1726_ < 1536u) {
                                            } else {
                                                phi_17809_ = true;
                                                break;
                                            }
                                            let _e1729_ = scan_scratch[_e1726_];
                                            let _e1730_ = (256u + _e1726_);
                                            if (_e1730_ < 1536u) {
                                            } else {
                                                phi_17809_ = true;
                                                break;
                                            }
                                            let _e1733_ = scan_scratch[_e1730_];
                                            let _e1734_ = (512u + _e1726_);
                                            if (_e1734_ < 1536u) {
                                            } else {
                                                phi_17809_ = true;
                                                break;
                                            }
                                            let _e1737_ = scan_scratch[_e1734_];
                                            let _e1738_ = (768u + _e1726_);
                                            if (_e1738_ < 1536u) {
                                            } else {
                                                phi_17809_ = true;
                                                break;
                                            }
                                            let _e1741_ = scan_scratch[_e1738_];
                                            let _e1742_ = (1024u + _e1726_);
                                            if (_e1742_ < 1536u) {
                                            } else {
                                                phi_17809_ = true;
                                                break;
                                            }
                                            let _e1745_ = scan_scratch[_e1742_];
                                            let _e1746_ = (1280u + _e1726_);
                                            if (_e1746_ < 1536u) {
                                            } else {
                                                phi_17809_ = true;
                                                break;
                                            }
                                            let _e1749_ = scan_scratch[_e1746_];
                                            phi_4177_ = type_26_(((_e1722_.member * _e1729_) + (_e1722_.member_1_ * _e1737_)), ((_e1722_.member * _e1733_) + (_e1722_.member_1_ * _e1741_)), ((_e1722_.member_2_ * _e1729_) + (_e1722_.member_3_ * _e1737_)), ((_e1722_.member_2_ * _e1733_) + (_e1722_.member_3_ * _e1741_)), (((_e1722_.member * _e1745_) + (_e1722_.member_1_ * _e1749_)) + _e1722_.member_4_), (((_e1722_.member_2_ * _e1745_) + (_e1722_.member_3_ * _e1749_)) + _e1722_.member_5_));
                                        } else {
                                            phi_4177_ = _e1722_;
                                        }
                                        let _e1778_ = phi_4177_;
                                        workgroupBarrier();
                                        if _e1697_ {
                                        } else {
                                            phi_17809_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1695_] = _e1778_.member;
                                        if _e1701_ {
                                        } else {
                                            phi_17809_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1700_] = _e1778_.member_1_;
                                        if _e1705_ {
                                        } else {
                                            phi_17809_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1704_] = _e1778_.member_2_;
                                        if _e1709_ {
                                        } else {
                                            phi_17809_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1708_] = _e1778_.member_3_;
                                        if _e1713_ {
                                        } else {
                                            phi_17809_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1712_] = _e1778_.member_4_;
                                        if _e1717_ {
                                        } else {
                                            phi_17809_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1716_] = _e1778_.member_5_;
                                        workgroupBarrier();
                                        phi_4079_ = (_e1720_ << bitcast<u32>(1i));
                                        phi_4082_ = _e1778_;
                                    } else {
                                        phi_4079_ = u32();
                                        phi_4082_ = type_26_();
                                    }
                                    let _e1788_ = phi_4079_;
                                    let _e1790_ = phi_4082_;
                                    phi_4078_ = _e1788_;
                                    phi_4081_ = _e1790_;
                                    phi_17809_ = _e1340_;
                                    if !(_e1723_) {
                                        break;
                                    }
                                }
                                let _e1793_ = phi_17809_;
                                phi_18498_ = _e1793_;
                                if _e1793_ {
                                    break;
                                }
                                let _e1794_ = (_e1694_ == 0u);
                                if _e1794_ {
                                    phi_4289_ = _e1693_;
                                } else {
                                    let _e1796_ = ((_e1694_ - 1u) & 255u);
                                    if (_e1796_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1799_ = scan_scratch[_e1796_];
                                    let _e1800_ = (256u + _e1796_);
                                    if (_e1800_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1803_ = scan_scratch[_e1800_];
                                    let _e1804_ = (512u + _e1796_);
                                    if (_e1804_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1807_ = scan_scratch[_e1804_];
                                    let _e1808_ = (768u + _e1796_);
                                    if (_e1808_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1811_ = scan_scratch[_e1808_];
                                    let _e1812_ = (1024u + _e1796_);
                                    if (_e1812_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1815_ = scan_scratch[_e1812_];
                                    let _e1816_ = (1280u + _e1796_);
                                    if (_e1816_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1819_ = scan_scratch[_e1816_];
                                    phi_4289_ = array<f32, 2>((((_e1799_ * _e1685_) + (_e1803_ * _e1692_)) + _e1815_), (((_e1807_ * _e1685_) + (_e1811_ * _e1692_)) + _e1819_));
                                }
                                let _e1830_ = phi_4289_;
                                let _e1833_ = (_e946_ > 0u);
                                if _e1833_ {
                                    if (_e946_ <= 256u) {
                                        phi_4301_ = (_e946_ - 1u);
                                    } else {
                                        phi_4301_ = 255u;
                                    }
                                    let _e1837_ = phi_4301_;
                                    let _e1838_ = (_e1837_ & 255u);
                                    if (_e1838_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1841_ = scan_scratch[_e1838_];
                                    let _e1842_ = (256u + _e1838_);
                                    if (_e1842_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1845_ = scan_scratch[_e1842_];
                                    let _e1846_ = (512u + _e1838_);
                                    if (_e1846_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1849_ = scan_scratch[_e1846_];
                                    let _e1850_ = (768u + _e1838_);
                                    if (_e1850_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1853_ = scan_scratch[_e1850_];
                                    let _e1854_ = (1024u + _e1838_);
                                    if (_e1854_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1857_ = scan_scratch[_e1854_];
                                    let _e1858_ = (1280u + _e1838_);
                                    if (_e1858_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e1861_ = scan_scratch[_e1858_];
                                    phi_4359_ = array<f32, 2>((((_e1841_ * _e1685_) + (_e1845_ * _e1692_)) + _e1857_), (((_e1849_ * _e1685_) + (_e1853_ * _e1692_)) + _e1861_));
                                } else {
                                    phi_4359_ = _e1693_;
                                }
                                let _e1872_ = phi_4359_;
                                workgroupBarrier();
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_2_[_e1364_] = _e1872_[0];
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_4_[_e1364_] = _e1872_[1];
                                let _e1877_ = (_e1649_ - _e1830_[1]);
                                let _e1880_ = ((_e1413_ * _e1830_[0]) + (_e1420_ * _e1877_));
                                let _e1884_ = ((_e1830_[1] + (_e1420_ * _e1830_[0])) + (_e1427_ * _e1877_));
                                let _e1899_ = (_e120_ < 256u);
                                if _e1899_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                aux_scratch[_e120_] = (((((_e1884_ * _e1448_) + (_e1880_ * _e1455_)) + (((_e1649_ - (_e1434_ * _e1880_)) - _e1884_) * _e1462_)) + ((_e1649_ - (_e1441_ * _e1880_)) * _e1469_)) + (_e1649_ * _e1476_));
                                if _e1304_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                switch bitcast<i32>(_e1364_) {
                                    case 0: {
                                        lane_buf[_e120_].x = _e1880_;
                                        break;
                                    }
                                    case 1: {
                                        lane_buf[_e120_].y = _e1880_;
                                        break;
                                    }
                                    case 2: {
                                        lane_buf[_e120_].z = _e1880_;
                                        break;
                                    }
                                    default: {
                                        lane_buf[_e120_].w = _e1880_;
                                        break;
                                    }
                                }
                                if _e948_ {
                                    let _e1910_ = (_e1420_ + _e1420_);
                                    let _e1911_ = (_e1427_ + _e1427_);
                                    phi_4620_ = type_26_(((_e1413_ + _e1413_) - 1f), -(_e1910_), _e1910_, (1f - _e1911_), (_e1910_ * _e1666_), (_e1911_ * _e1666_));
                                } else {
                                    phi_4620_ = type_26_(1f, 0f, 0f, 1f, 0f, 0f);
                                }
                                let _e1920_ = phi_4620_;
                                local_12_ = array<f32, 4>(_e920_.member.member_1_.member[0], _e920_.member.member_1_.member[1], _e920_.member.member_1_.member[2], _e920_.member.member_1_.member[3]);
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e1927_ = local_12_[_e1364_];
                                local_13_ = array<f32, 4>(_e920_.member_1_.member_1_.member[0], _e920_.member_1_.member_1_.member[1], _e920_.member_1_.member_1_.member[2], _e920_.member_1_.member_1_.member[3]);
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e1934_ = local_13_[_e1364_];
                                let _e1935_ = array<f32, 2>(_e1927_, _e1934_);
                                if _e1697_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1695_] = _e1920_.member;
                                if _e1701_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1700_] = _e1920_.member_1_;
                                if _e1705_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1704_] = _e1920_.member_2_;
                                if _e1709_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1708_] = _e1920_.member_3_;
                                if _e1713_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1712_] = _e1920_.member_4_;
                                if _e1717_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1716_] = _e1920_.member_5_;
                                workgroupBarrier();
                                phi_4698_ = 1u;
                                phi_4701_ = _e1920_;
                                loop {
                                    let _e1943_ = phi_4698_;
                                    let _e1945_ = phi_4701_;
                                    let _e1946_ = (_e1943_ < 256u);
                                    if _e1946_ {
                                        if (_e1694_ >= _e1943_) {
                                            let _e1949_ = ((_e1694_ - _e1943_) & 255u);
                                            if (_e1949_ < 1536u) {
                                            } else {
                                                phi_18124_ = true;
                                                break;
                                            }
                                            let _e1952_ = scan_scratch[_e1949_];
                                            let _e1953_ = (256u + _e1949_);
                                            if (_e1953_ < 1536u) {
                                            } else {
                                                phi_18124_ = true;
                                                break;
                                            }
                                            let _e1956_ = scan_scratch[_e1953_];
                                            let _e1957_ = (512u + _e1949_);
                                            if (_e1957_ < 1536u) {
                                            } else {
                                                phi_18124_ = true;
                                                break;
                                            }
                                            let _e1960_ = scan_scratch[_e1957_];
                                            let _e1961_ = (768u + _e1949_);
                                            if (_e1961_ < 1536u) {
                                            } else {
                                                phi_18124_ = true;
                                                break;
                                            }
                                            let _e1964_ = scan_scratch[_e1961_];
                                            let _e1965_ = (1024u + _e1949_);
                                            if (_e1965_ < 1536u) {
                                            } else {
                                                phi_18124_ = true;
                                                break;
                                            }
                                            let _e1968_ = scan_scratch[_e1965_];
                                            let _e1969_ = (1280u + _e1949_);
                                            if (_e1969_ < 1536u) {
                                            } else {
                                                phi_18124_ = true;
                                                break;
                                            }
                                            let _e1972_ = scan_scratch[_e1969_];
                                            phi_4797_ = type_26_(((_e1945_.member * _e1952_) + (_e1945_.member_1_ * _e1960_)), ((_e1945_.member * _e1956_) + (_e1945_.member_1_ * _e1964_)), ((_e1945_.member_2_ * _e1952_) + (_e1945_.member_3_ * _e1960_)), ((_e1945_.member_2_ * _e1956_) + (_e1945_.member_3_ * _e1964_)), (((_e1945_.member * _e1968_) + (_e1945_.member_1_ * _e1972_)) + _e1945_.member_4_), (((_e1945_.member_2_ * _e1968_) + (_e1945_.member_3_ * _e1972_)) + _e1945_.member_5_));
                                        } else {
                                            phi_4797_ = _e1945_;
                                        }
                                        let _e2001_ = phi_4797_;
                                        workgroupBarrier();
                                        if _e1697_ {
                                        } else {
                                            phi_18124_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1695_] = _e2001_.member;
                                        if _e1701_ {
                                        } else {
                                            phi_18124_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1700_] = _e2001_.member_1_;
                                        if _e1705_ {
                                        } else {
                                            phi_18124_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1704_] = _e2001_.member_2_;
                                        if _e1709_ {
                                        } else {
                                            phi_18124_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1708_] = _e2001_.member_3_;
                                        if _e1713_ {
                                        } else {
                                            phi_18124_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1712_] = _e2001_.member_4_;
                                        if _e1717_ {
                                        } else {
                                            phi_18124_ = true;
                                            break;
                                        }
                                        scan_scratch[_e1716_] = _e2001_.member_5_;
                                        workgroupBarrier();
                                        phi_4699_ = (_e1943_ << bitcast<u32>(1i));
                                        phi_4702_ = _e2001_;
                                    } else {
                                        phi_4699_ = u32();
                                        phi_4702_ = type_26_();
                                    }
                                    let _e2011_ = phi_4699_;
                                    let _e2013_ = phi_4702_;
                                    phi_4698_ = _e2011_;
                                    phi_4701_ = _e2013_;
                                    phi_18124_ = _e1793_;
                                    if !(_e1946_) {
                                        break;
                                    }
                                }
                                let _e2016_ = phi_18124_;
                                phi_18498_ = _e2016_;
                                if _e2016_ {
                                    break;
                                }
                                if _e1794_ {
                                    phi_4909_ = _e1935_;
                                } else {
                                    let _e2018_ = ((_e1694_ - 1u) & 255u);
                                    if (_e2018_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2021_ = scan_scratch[_e2018_];
                                    let _e2022_ = (256u + _e2018_);
                                    if (_e2022_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2025_ = scan_scratch[_e2022_];
                                    let _e2026_ = (512u + _e2018_);
                                    if (_e2026_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2029_ = scan_scratch[_e2026_];
                                    let _e2030_ = (768u + _e2018_);
                                    if (_e2030_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2033_ = scan_scratch[_e2030_];
                                    let _e2034_ = (1024u + _e2018_);
                                    if (_e2034_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2037_ = scan_scratch[_e2034_];
                                    let _e2038_ = (1280u + _e2018_);
                                    if (_e2038_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2041_ = scan_scratch[_e2038_];
                                    phi_4909_ = array<f32, 2>((((_e2021_ * _e1927_) + (_e2025_ * _e1934_)) + _e2037_), (((_e2029_ * _e1927_) + (_e2033_ * _e1934_)) + _e2041_));
                                }
                                let _e2052_ = phi_4909_;
                                if _e1833_ {
                                    if (_e946_ <= 256u) {
                                        phi_4921_ = (_e946_ - 1u);
                                    } else {
                                        phi_4921_ = 255u;
                                    }
                                    let _e2058_ = phi_4921_;
                                    let _e2059_ = (_e2058_ & 255u);
                                    if (_e2059_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2062_ = scan_scratch[_e2059_];
                                    let _e2063_ = (256u + _e2059_);
                                    if (_e2063_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2066_ = scan_scratch[_e2063_];
                                    let _e2067_ = (512u + _e2059_);
                                    if (_e2067_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2070_ = scan_scratch[_e2067_];
                                    let _e2071_ = (768u + _e2059_);
                                    if (_e2071_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2074_ = scan_scratch[_e2071_];
                                    let _e2075_ = (1024u + _e2059_);
                                    if (_e2075_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2078_ = scan_scratch[_e2075_];
                                    let _e2079_ = (1280u + _e2059_);
                                    if (_e2079_ < 1536u) {
                                    } else {
                                        phi_18498_ = true;
                                        break;
                                    }
                                    let _e2082_ = scan_scratch[_e2079_];
                                    phi_4979_ = array<f32, 2>((((_e2062_ * _e1927_) + (_e2066_ * _e1934_)) + _e2078_), (((_e2070_ * _e1927_) + (_e2074_ * _e1934_)) + _e2082_));
                                } else {
                                    phi_4979_ = _e1935_;
                                }
                                let _e2093_ = phi_4979_;
                                workgroupBarrier();
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_3_[_e1364_] = _e2093_[0];
                                if _e1683_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                local_5_[_e1364_] = _e2093_[1];
                                let _e2098_ = (_e1666_ - _e2052_[1]);
                                let _e2101_ = ((_e1413_ * _e2052_[0]) + (_e1420_ * _e2098_));
                                let _e2105_ = ((_e2052_[1] + (_e1420_ * _e2052_[0])) + (_e1427_ * _e2098_));
                                let _e2119_ = (((((_e2105_ * _e1448_) + (_e2101_ * _e1455_)) + (((_e1666_ - (_e1434_ * _e2101_)) - _e2105_) * _e1462_)) + ((_e1666_ - (_e1441_ * _e2101_)) * _e1469_)) + (_e1666_ * _e1476_));
                                if _e1899_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2120_ = aux_scratch[_e120_];
                                if _e1304_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2121_ = lane_buf[_e120_];
                                switch bitcast<i32>(_e1364_) {
                                    case 0: {
                                        phi_5229_ = _e2121_.x;
                                        break;
                                    }
                                    case 1: {
                                        phi_5229_ = _e2121_.y;
                                        break;
                                    }
                                    case 2: {
                                        phi_5229_ = _e2121_.z;
                                        break;
                                    }
                                    default: {
                                        phi_5229_ = _e2121_.w;
                                        break;
                                    }
                                }
                                let _e2128_ = phi_5229_;
                                if (_e120_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e120_] = _e2120_;
                                if (_e1306_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e1306_] = _e2128_;
                                let _e2133_ = (512u + _e120_);
                                if (_e2133_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2133_] = _e2119_;
                                let _e2136_ = (768u + _e120_);
                                if (_e2136_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2136_] = _e2101_;
                                let _e2139_ = (1024u + _e120_);
                                if (_e2139_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2139_] = 0f;
                                let _e2142_ = (1280u + _e120_);
                                if (_e2142_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                scan_scratch[_e2142_] = 0f;
                                workgroupBarrier();
                                phi_18218_ = _e2016_;
                                phi_5271_ = _e2120_;
                                phi_5272_ = _e2119_;
                            }
                            let _e2494_ = phi_18218_;
                            let _e2496_ = phi_5271_;
                            let _e2498_ = phi_5272_;
                            if _e949_ {
                                local_14_ = array<f32, 4>(_e1344_.member_2_.member.member[0], _e1344_.member_2_.member.member[1], _e1344_.member_2_.member.member[2], _e1344_.member_2_.member.member[3]);
                                local_15_ = array<f32, 4>(_e1344_.member_2_.member_1_.member[0], _e1344_.member_2_.member_1_.member[1], _e1344_.member_2_.member_1_.member[2], _e1344_.member_2_.member_1_.member[3]);
                                let _e2514_ = (_e1364_ < 4u);
                                if _e2514_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2516_ = local_14_[_e1364_];
                                if _e2514_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2518_ = local_15_[_e1364_];
                                phi_5324_ = type_16_(0u, _e946_);
                                phi_5327_ = _e2518_;
                                phi_5329_ = _e2516_;
                                loop {
                                    let _e2521_ = phi_5324_;
                                    let _e2523_ = phi_5327_;
                                    let _e2525_ = phi_5329_;
                                    local_21_ = _e2525_;
                                    local_22_ = _e2523_;
                                    if (_e2521_.member < _e2521_.member_1_) {
                                        phi_5325_ = type_16_((_e2521_.member + 1u), _e2521_.member_1_);
                                        phi_5352_ = type_16_(1u, _e2521_.member);
                                    } else {
                                        phi_5325_ = _e2521_;
                                        phi_5352_ = type_16_(0u, u32());
                                    }
                                    let _e2538_ = phi_5325_;
                                    let _e2540_ = phi_5352_;
                                    let _e2544_ = (bitcast<i32>(_e2540_.member) != 0i);
                                    if _e2544_ {
                                        let _e2545_ = (_e2540_.member_1_ & 255u);
                                        if (_e2545_ < 1536u) {
                                        } else {
                                            phi_18214_ = true;
                                            break;
                                        }
                                        let _e2548_ = scan_scratch[_e2545_];
                                        let _e2549_ = (256u + _e2545_);
                                        if (_e2549_ < 1536u) {
                                        } else {
                                            phi_18214_ = true;
                                            break;
                                        }
                                        let _e2552_ = scan_scratch[_e2549_];
                                        let _e2553_ = (512u + _e2545_);
                                        if (_e2553_ < 1536u) {
                                        } else {
                                            phi_18214_ = true;
                                            break;
                                        }
                                        let _e2556_ = scan_scratch[_e2553_];
                                        let _e2557_ = (768u + _e2545_);
                                        if (_e2557_ < 1536u) {
                                        } else {
                                            phi_18214_ = true;
                                            break;
                                        }
                                        let _e2560_ = scan_scratch[_e2557_];
                                        let _e2563_ = bitcast<u32>((_e1597_ * _e2552_));
                                        let _e2564_ = (_e2563_ & 2139095040u);
                                        if (_e2564_ == 2139095040u) {
                                            phi_5406_ = 0u;
                                        } else {
                                            if (_e2564_ == 0u) {
                                                phi_5405_ = (_e2563_ & 2147483648u);
                                            } else {
                                                phi_5405_ = _e2563_;
                                            }
                                            let _e2569_ = phi_5405_;
                                            phi_5406_ = _e2569_;
                                        }
                                        let _e2571_ = phi_5406_;
                                        let _e2577_ = (_e1552_ <= 0f);
                                        let _e2579_ = (1f - _e1514_);
                                        let _e2580_ = (_e2579_ + select((_e1552_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2571_)) & 2147483647u))), 0f, _e2577_));
                                        if (_e2580_ != _e2580_) {
                                            phi_12509_ = true;
                                        } else {
                                            phi_12509_ = (0f >= _e2580_);
                                        }
                                        let _e2584_ = phi_12509_;
                                        let _e2585_ = select(_e2580_, 0f, _e2584_);
                                        if (_e2585_ != _e2585_) {
                                            phi_12569_ = true;
                                        } else {
                                            phi_12569_ = (1f <= _e2585_);
                                        }
                                        let _e2589_ = phi_12569_;
                                        let _e2590_ = select(_e2585_, 1f, _e2589_);
                                        let _e2591_ = (_e1495_ * _e2525_);
                                        let _e2592_ = (1f - _e1571_);
                                        let _e2593_ = (_e1533_ * _e2592_);
                                        let _e2595_ = ((_e1604_ * _e2548_) - (_e2593_ * _e2591_));
                                        if (_e2595_ != _e2595_) {
                                            phi_12629_ = true;
                                        } else {
                                            phi_12629_ = (-1f >= _e2595_);
                                        }
                                        let _e2599_ = phi_12629_;
                                        let _e2600_ = select(_e2595_, -1f, _e2599_);
                                        if (_e2600_ != _e2600_) {
                                            phi_12689_ = true;
                                        } else {
                                            phi_12689_ = (1f <= _e2600_);
                                        }
                                        let _e2604_ = phi_12689_;
                                        let _e2605_ = select(_e2600_, 1f, _e2604_);
                                        let _e2612_ = bitcast<f32>((bitcast<u32>(_e2595_) & 2147483647u));
                                        if (_e2612_ != _e2612_) {
                                            phi_12749_ = true;
                                        } else {
                                            phi_12749_ = (1000000000000000f <= _e2612_);
                                        }
                                        let _e2616_ = phi_12749_;
                                        let _e2617_ = select(_e2612_, 1000000000000000f, _e2616_);
                                        let _e2618_ = (_e2617_ * _e2617_);
                                        let _e2619_ = (_e2618_ + 0.0000010000001f);
                                        if (_e2619_ != _e2619_) {
                                            phi_12809_ = true;
                                        } else {
                                            phi_12809_ = (0.000000000001f >= _e2619_);
                                        }
                                        let _e2623_ = phi_12809_;
                                        let _e2624_ = select(_e2619_, 0.000000000001f, _e2623_);
                                        let _e2625_ = (_e2624_ * 0.5f);
                                        let _e2630_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2624_) >> bitcast<u32>(1i))));
                                        let _e2634_ = (_e2630_ * (1.5f - (_e2625_ * (_e2630_ * _e2630_))));
                                        let _e2638_ = (_e2634_ * (1.5f - (_e2625_ * (_e2634_ * _e2634_))));
                                        let _e2643_ = (_e2619_ * (_e2638_ * (1.5f - (_e2625_ * (_e2638_ * _e2638_)))));
                                        let _e2644_ = (_e2643_ + 0.001f);
                                        if (_e2644_ != _e2644_) {
                                            phi_12869_ = true;
                                        } else {
                                            phi_12869_ = (0.000001f >= _e2644_);
                                        }
                                        let _e2648_ = phi_12869_;
                                        let _e2649_ = select(_e2644_, 0.000001f, _e2648_);
                                        let _e2652_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2649_)));
                                        let _e2656_ = (_e2652_ + (_e2652_ * (1f - (_e2649_ * _e2652_))));
                                        let _e2660_ = (_e2656_ + (_e2656_ * (1f - (_e2649_ * _e2656_))));
                                        let _e2664_ = (_e2660_ + (_e2660_ * (1f - (_e2649_ * _e2660_))));
                                        let _e2665_ = (_e2643_ + _e2617_);
                                        if (_e2665_ != _e2665_) {
                                            phi_12929_ = true;
                                        } else {
                                            phi_12929_ = (0.000001f >= _e2665_);
                                        }
                                        let _e2669_ = phi_12929_;
                                        let _e2670_ = select(_e2665_, 0.000001f, _e2669_);
                                        let _e2673_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2670_)));
                                        let _e2677_ = (_e2673_ + (_e2673_ * (1f - (_e2670_ * _e2673_))));
                                        let _e2681_ = (_e2677_ + (_e2677_ * (1f - (_e2670_ * _e2677_))));
                                        let _e2691_ = (_e2612_ <= 1000000000000000f);
                                        let _e2701_ = select(select(select(_e2612_, (0.5f * (_e2617_ + (_e2618_ * _e2664_))), _e2691_), select(-0.0005f, -(((0.0005f * (_e2617_ * (_e2681_ + (_e2681_ * (1f - (_e2670_ * _e2681_)))))) * (1f + (_e2617_ * _e2664_)))), _e2691_), (_e2595_ <= 0f)), 0f, (_e2612_ <= 0f));
                                        let _e2705_ = (_e1571_ <= 0f);
                                        let _e2712_ = bitcast<u32>((((1f - (_e1571_ * _e2590_)) * _e2591_) + (_e2590_ * select(((_e2592_ * _e2701_) + (_e1571_ * (_e2605_ * (1f - (0.33333334f * (_e2605_ * _e2605_)))))), _e2701_, _e2705_))));
                                        let _e2713_ = (_e2712_ & 2139095040u);
                                        if (_e2713_ == 2139095040u) {
                                            phi_6148_ = 0u;
                                        } else {
                                            if (_e2713_ == 0u) {
                                                phi_6147_ = (_e2712_ & 2147483648u);
                                            } else {
                                                phi_6147_ = _e2712_;
                                            }
                                            let _e2718_ = phi_6147_;
                                            phi_6148_ = _e2718_;
                                        }
                                        let _e2720_ = phi_6148_;
                                        let _e2721_ = bitcast<f32>(_e2720_);
                                        let _e2724_ = bitcast<u32>((_e1597_ * _e2560_));
                                        let _e2725_ = (_e2724_ & 2139095040u);
                                        if (_e2725_ == 2139095040u) {
                                            phi_6210_ = 0u;
                                        } else {
                                            if (_e2725_ == 0u) {
                                                phi_6209_ = (_e2724_ & 2147483648u);
                                            } else {
                                                phi_6209_ = _e2724_;
                                            }
                                            let _e2730_ = phi_6209_;
                                            phi_6210_ = _e2730_;
                                        }
                                        let _e2732_ = phi_6210_;
                                        let _e2739_ = (_e2579_ + select((_e1552_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2732_)) & 2147483647u))), 0f, _e2577_));
                                        if (_e2739_ != _e2739_) {
                                            phi_13109_ = true;
                                        } else {
                                            phi_13109_ = (0f >= _e2739_);
                                        }
                                        let _e2743_ = phi_13109_;
                                        let _e2744_ = select(_e2739_, 0f, _e2743_);
                                        if (_e2744_ != _e2744_) {
                                            phi_13169_ = true;
                                        } else {
                                            phi_13169_ = (1f <= _e2744_);
                                        }
                                        let _e2748_ = phi_13169_;
                                        let _e2749_ = select(_e2744_, 1f, _e2748_);
                                        let _e2750_ = (_e1495_ * _e2523_);
                                        let _e2752_ = ((_e1604_ * _e2556_) - (_e2593_ * _e2750_));
                                        if (_e2752_ != _e2752_) {
                                            phi_13229_ = true;
                                        } else {
                                            phi_13229_ = (-1f >= _e2752_);
                                        }
                                        let _e2756_ = phi_13229_;
                                        let _e2757_ = select(_e2752_, -1f, _e2756_);
                                        if (_e2757_ != _e2757_) {
                                            phi_13289_ = true;
                                        } else {
                                            phi_13289_ = (1f <= _e2757_);
                                        }
                                        let _e2761_ = phi_13289_;
                                        let _e2762_ = select(_e2757_, 1f, _e2761_);
                                        let _e2769_ = bitcast<f32>((bitcast<u32>(_e2752_) & 2147483647u));
                                        if (_e2769_ != _e2769_) {
                                            phi_13349_ = true;
                                        } else {
                                            phi_13349_ = (1000000000000000f <= _e2769_);
                                        }
                                        let _e2773_ = phi_13349_;
                                        let _e2774_ = select(_e2769_, 1000000000000000f, _e2773_);
                                        let _e2775_ = (_e2774_ * _e2774_);
                                        let _e2776_ = (_e2775_ + 0.0000010000001f);
                                        if (_e2776_ != _e2776_) {
                                            phi_13409_ = true;
                                        } else {
                                            phi_13409_ = (0.000000000001f >= _e2776_);
                                        }
                                        let _e2780_ = phi_13409_;
                                        let _e2781_ = select(_e2776_, 0.000000000001f, _e2780_);
                                        let _e2782_ = (_e2781_ * 0.5f);
                                        let _e2787_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2781_) >> bitcast<u32>(1i))));
                                        let _e2791_ = (_e2787_ * (1.5f - (_e2782_ * (_e2787_ * _e2787_))));
                                        let _e2795_ = (_e2791_ * (1.5f - (_e2782_ * (_e2791_ * _e2791_))));
                                        let _e2800_ = (_e2776_ * (_e2795_ * (1.5f - (_e2782_ * (_e2795_ * _e2795_)))));
                                        let _e2801_ = (_e2800_ + 0.001f);
                                        if (_e2801_ != _e2801_) {
                                            phi_13469_ = true;
                                        } else {
                                            phi_13469_ = (0.000001f >= _e2801_);
                                        }
                                        let _e2805_ = phi_13469_;
                                        let _e2806_ = select(_e2801_, 0.000001f, _e2805_);
                                        let _e2809_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2806_)));
                                        let _e2813_ = (_e2809_ + (_e2809_ * (1f - (_e2806_ * _e2809_))));
                                        let _e2817_ = (_e2813_ + (_e2813_ * (1f - (_e2806_ * _e2813_))));
                                        let _e2821_ = (_e2817_ + (_e2817_ * (1f - (_e2806_ * _e2817_))));
                                        let _e2822_ = (_e2800_ + _e2774_);
                                        if (_e2822_ != _e2822_) {
                                            phi_13529_ = true;
                                        } else {
                                            phi_13529_ = (0.000001f >= _e2822_);
                                        }
                                        let _e2826_ = phi_13529_;
                                        let _e2827_ = select(_e2822_, 0.000001f, _e2826_);
                                        let _e2830_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2827_)));
                                        let _e2834_ = (_e2830_ + (_e2830_ * (1f - (_e2827_ * _e2830_))));
                                        let _e2838_ = (_e2834_ + (_e2834_ * (1f - (_e2827_ * _e2834_))));
                                        let _e2848_ = (_e2769_ <= 1000000000000000f);
                                        let _e2858_ = select(select(select(_e2769_, (0.5f * (_e2774_ + (_e2775_ * _e2821_))), _e2848_), select(-0.0005f, -(((0.0005f * (_e2774_ * (_e2838_ + (_e2838_ * (1f - (_e2827_ * _e2838_)))))) * (1f + (_e2774_ * _e2821_)))), _e2848_), (_e2752_ <= 0f)), 0f, (_e2769_ <= 0f));
                                        let _e2868_ = bitcast<u32>((((1f - (_e1571_ * _e2749_)) * _e2750_) + (_e2749_ * select(((_e2592_ * _e2858_) + (_e1571_ * (_e2762_ * (1f - (0.33333334f * (_e2762_ * _e2762_)))))), _e2858_, _e2705_))));
                                        let _e2869_ = (_e2868_ & 2139095040u);
                                        if (_e2869_ == 2139095040u) {
                                            phi_6952_ = 0u;
                                        } else {
                                            if (_e2869_ == 0u) {
                                                phi_6951_ = (_e2868_ & 2147483648u);
                                            } else {
                                                phi_6951_ = _e2868_;
                                            }
                                            let _e2874_ = phi_6951_;
                                            phi_6952_ = _e2874_;
                                        }
                                        let _e2876_ = phi_6952_;
                                        let _e2877_ = bitcast<f32>(_e2876_);
                                        let _e2879_ = (0.5f * (_e2721_ + _e2877_));
                                        let _e2886_ = (1024u + _e2545_);
                                        if (_e2886_ < 1536u) {
                                        } else {
                                            phi_18214_ = true;
                                            break;
                                        }
                                        scan_scratch[_e2886_] = (_e2721_ + (_e1590_ * (_e2879_ - _e2721_)));
                                        let _e2889_ = (1280u + _e2545_);
                                        if (_e2889_ < 1536u) {
                                        } else {
                                            phi_18214_ = true;
                                            break;
                                        }
                                        scan_scratch[_e2889_] = (_e2877_ + (_e1590_ * (_e2879_ - _e2877_)));
                                        phi_5328_ = _e2877_;
                                        phi_5330_ = _e2721_;
                                    } else {
                                        phi_5328_ = f32();
                                        phi_5330_ = f32();
                                    }
                                    let _e2893_ = phi_5328_;
                                    let _e2895_ = phi_5330_;
                                    phi_5324_ = _e2538_;
                                    phi_5327_ = _e2893_;
                                    phi_5329_ = _e2895_;
                                    phi_18214_ = _e2494_;
                                    if !(_e2544_) {
                                        break;
                                    }
                                }
                                let _e2898_ = phi_18214_;
                                phi_18498_ = _e2898_;
                                if _e2898_ {
                                    break;
                                }
                                if _e2514_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2900_ = local_21_;
                                local_14_[_e1364_] = _e2900_;
                                if _e2514_ {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2902_ = local_22_;
                                local_15_[_e1364_] = _e2902_;
                                let _e2903_ = local_14_;
                                let _e2909_ = local_15_;
                                phi_18510_ = _e2898_;
                                phi_7094_ = type_32_(_e1344_.member, _e1344_.member_1_, type_27_(type_24_(array<f32, 4>(_e2903_[0], _e2903_[1], _e2903_[2], _e2903_[3])), type_24_(array<f32, 4>(_e2909_[0], _e2909_[1], _e2909_[2], _e2909_[3]))), _e1344_.member_3_);
                            } else {
                                phi_18510_ = _e2494_;
                                phi_7094_ = _e1344_;
                            }
                            let _e2924_ = phi_18510_;
                            let _e2926_ = phi_7094_;
                            workgroupBarrier();
                            if _e948_ {
                                let _e2927_ = (1024u + _e120_);
                                if (_e2927_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2930_ = scan_scratch[_e2927_];
                                phi_7106_ = _e2930_;
                            } else {
                                phi_7106_ = 0f;
                            }
                            let _e2932_ = phi_7106_;
                            if _e948_ {
                                let _e2933_ = (1280u + _e120_);
                                if (_e2933_ < 1536u) {
                                } else {
                                    phi_18498_ = true;
                                    break;
                                }
                                let _e2936_ = scan_scratch[_e2933_];
                                phi_7118_ = _e2936_;
                            } else {
                                phi_7118_ = 0f;
                            }
                            let _e2938_ = phi_7118_;
                            let _e2939_ = (_e1364_ < 4u);
                            if _e2939_ {
                            } else {
                                phi_18498_ = true;
                                break;
                            }
                            local[_e1364_] = _e2932_;
                            if _e2939_ {
                            } else {
                                phi_18498_ = true;
                                break;
                            }
                            local_1_[_e1364_] = _e2938_;
                            let _e2943_ = (1f - (_e1611_ * _e2932_));
                            if (_e2943_ != _e2943_) {
                                phi_13709_ = true;
                            } else {
                                phi_13709_ = (0.05f >= _e2943_);
                            }
                            let _e2947_ = phi_13709_;
                            let _e2950_ = (1f - (_e1611_ * _e2938_));
                            if (_e2950_ != _e2950_) {
                                phi_13724_ = true;
                            } else {
                                phi_13724_ = (0.05f >= _e2950_);
                            }
                            let _e2954_ = phi_13724_;
                            if _e948_ {
                                let _e2960_ = bitcast<u32>((((_e2496_ * _e1618_) * select(_e2943_, 0.05f, _e2947_)) + (_e1625_ * _e2932_)));
                                let _e2961_ = (_e2960_ & 2139095040u);
                                if (_e2961_ == 2139095040u) {
                                    phi_7156_ = true;
                                } else {
                                    phi_7156_ = (_e2961_ == 0u);
                                }
                                let _e2965_ = phi_7156_;
                                phi_7162_ = bitcast<f32>(select(_e2960_, 0u, _e2965_));
                            } else {
                                phi_7162_ = 0f;
                            }
                            let _e2969_ = phi_7162_;
                            if _e948_ {
                                let _e2974_ = bitcast<u32>((((_e2498_ * _e1618_) * select(_e2950_, 0.05f, _e2954_)) + (_e1625_ * _e2938_)));
                                let _e2975_ = (_e2974_ & 2139095040u);
                                if (_e2975_ == 2139095040u) {
                                    phi_7180_ = true;
                                } else {
                                    phi_7180_ = (_e2975_ == 0u);
                                }
                                let _e2979_ = phi_7180_;
                                phi_7186_ = bitcast<f32>(select(_e2974_, 0u, _e2979_));
                            } else {
                                phi_7186_ = 0f;
                            }
                            let _e2983_ = phi_7186_;
                            if _e1304_ {
                            } else {
                                phi_18498_ = true;
                                break;
                            }
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    lane_buf[_e120_].x = _e2969_;
                                    break;
                                }
                                case 1: {
                                    lane_buf[_e120_].y = _e2969_;
                                    break;
                                }
                                case 2: {
                                    lane_buf[_e120_].z = _e2969_;
                                    break;
                                }
                                default: {
                                    lane_buf[_e120_].w = _e2969_;
                                    break;
                                }
                            }
                            if _e1307_ {
                            } else {
                                phi_18498_ = true;
                                break;
                            }
                            switch bitcast<i32>(_e1364_) {
                                case 0: {
                                    lane_buf[_e1306_].x = _e2983_;
                                    break;
                                }
                                case 1: {
                                    lane_buf[_e1306_].y = _e2983_;
                                    break;
                                }
                                case 2: {
                                    lane_buf[_e1306_].z = _e2983_;
                                    break;
                                }
                                default: {
                                    lane_buf[_e1306_].w = _e2983_;
                                    break;
                                }
                            }
                            workgroupBarrier();
                            phi_18499_ = _e2924_;
                            phi_2440_ = _e2926_;
                        } else {
                            phi_18499_ = _e1340_;
                            phi_2440_ = type_32_();
                        }
                        let _e3003_ = phi_18499_;
                        let _e3005_ = phi_2440_;
                        phi_17850_ = _e3003_;
                        phi_2436_ = _e1357_;
                        phi_2439_ = _e3005_;
                        phi_18498_ = _e3003_;
                        if !(_e1363_) {
                            break;
                        }
                    }
                    let _e3008_ = phi_18498_;
                    phi_18515_ = _e3008_;
                    if _e3008_ {
                        break;
                    }
                    let _e3009_ = local_2_;
                    let _e3014_ = bitcast<u32>(_e3009_[0]);
                    let _e3015_ = (_e3014_ & 2139095040u);
                    if (_e3015_ == 2139095040u) {
                        phi_7240_ = 0u;
                    } else {
                        if (_e3015_ == 0u) {
                            phi_7239_ = (_e3014_ & 2147483648u);
                        } else {
                            phi_7239_ = _e3014_;
                        }
                        let _e3020_ = phi_7239_;
                        phi_7240_ = _e3020_;
                    }
                    let _e3022_ = phi_7240_;
                    let _e3024_ = bitcast<u32>(_e3009_[1]);
                    let _e3025_ = (_e3024_ & 2139095040u);
                    if (_e3025_ == 2139095040u) {
                        phi_7255_ = 0u;
                    } else {
                        if (_e3025_ == 0u) {
                            phi_7254_ = (_e3024_ & 2147483648u);
                        } else {
                            phi_7254_ = _e3024_;
                        }
                        let _e3030_ = phi_7254_;
                        phi_7255_ = _e3030_;
                    }
                    let _e3032_ = phi_7255_;
                    let _e3034_ = bitcast<u32>(_e3009_[2]);
                    let _e3035_ = (_e3034_ & 2139095040u);
                    if (_e3035_ == 2139095040u) {
                        phi_7270_ = 0u;
                    } else {
                        if (_e3035_ == 0u) {
                            phi_7269_ = (_e3034_ & 2147483648u);
                        } else {
                            phi_7269_ = _e3034_;
                        }
                        let _e3040_ = phi_7269_;
                        phi_7270_ = _e3040_;
                    }
                    let _e3042_ = phi_7270_;
                    let _e3044_ = bitcast<u32>(_e3009_[3]);
                    let _e3045_ = (_e3044_ & 2139095040u);
                    if (_e3045_ == 2139095040u) {
                        phi_7285_ = 0u;
                    } else {
                        if (_e3045_ == 0u) {
                            phi_7284_ = (_e3044_ & 2147483648u);
                        } else {
                            phi_7284_ = _e3044_;
                        }
                        let _e3050_ = phi_7284_;
                        phi_7285_ = _e3050_;
                    }
                    let _e3052_ = phi_7285_;
                    let _e3055_ = local_3_;
                    let _e3060_ = bitcast<u32>(_e3055_[0]);
                    let _e3061_ = (_e3060_ & 2139095040u);
                    if (_e3061_ == 2139095040u) {
                        phi_7308_ = 0u;
                    } else {
                        if (_e3061_ == 0u) {
                            phi_7307_ = (_e3060_ & 2147483648u);
                        } else {
                            phi_7307_ = _e3060_;
                        }
                        let _e3066_ = phi_7307_;
                        phi_7308_ = _e3066_;
                    }
                    let _e3068_ = phi_7308_;
                    let _e3070_ = bitcast<u32>(_e3055_[1]);
                    let _e3071_ = (_e3070_ & 2139095040u);
                    if (_e3071_ == 2139095040u) {
                        phi_7323_ = 0u;
                    } else {
                        if (_e3071_ == 0u) {
                            phi_7322_ = (_e3070_ & 2147483648u);
                        } else {
                            phi_7322_ = _e3070_;
                        }
                        let _e3076_ = phi_7322_;
                        phi_7323_ = _e3076_;
                    }
                    let _e3078_ = phi_7323_;
                    let _e3080_ = bitcast<u32>(_e3055_[2]);
                    let _e3081_ = (_e3080_ & 2139095040u);
                    if (_e3081_ == 2139095040u) {
                        phi_7338_ = 0u;
                    } else {
                        if (_e3081_ == 0u) {
                            phi_7337_ = (_e3080_ & 2147483648u);
                        } else {
                            phi_7337_ = _e3080_;
                        }
                        let _e3086_ = phi_7337_;
                        phi_7338_ = _e3086_;
                    }
                    let _e3088_ = phi_7338_;
                    let _e3090_ = bitcast<u32>(_e3055_[3]);
                    let _e3091_ = (_e3090_ & 2139095040u);
                    if (_e3091_ == 2139095040u) {
                        phi_7353_ = 0u;
                    } else {
                        if (_e3091_ == 0u) {
                            phi_7352_ = (_e3090_ & 2147483648u);
                        } else {
                            phi_7352_ = _e3090_;
                        }
                        let _e3096_ = phi_7352_;
                        phi_7353_ = _e3096_;
                    }
                    let _e3098_ = phi_7353_;
                    let _e3105_ = local_23_;
                    let _e3110_ = type_32_(type_27_(type_24_(array<f32, 4>(bitcast<f32>(_e3022_), bitcast<f32>(_e3032_), bitcast<f32>(_e3042_), bitcast<f32>(_e3052_))), type_24_(array<f32, 4>(bitcast<f32>(_e3068_), bitcast<f32>(_e3078_), bitcast<f32>(_e3088_), bitcast<f32>(_e3098_)))), _e3105_.member_1_, _e3105_.member_2_, _e3105_.member_3_);
                    let _e3111_ = local_4_;
                    let _e3116_ = bitcast<u32>(_e3111_[0]);
                    let _e3117_ = (_e3116_ & 2139095040u);
                    if (_e3117_ == 2139095040u) {
                        phi_7379_ = 0u;
                    } else {
                        if (_e3117_ == 0u) {
                            phi_7378_ = (_e3116_ & 2147483648u);
                        } else {
                            phi_7378_ = _e3116_;
                        }
                        let _e3122_ = phi_7378_;
                        phi_7379_ = _e3122_;
                    }
                    let _e3124_ = phi_7379_;
                    let _e3126_ = bitcast<u32>(_e3111_[1]);
                    let _e3127_ = (_e3126_ & 2139095040u);
                    if (_e3127_ == 2139095040u) {
                        phi_7394_ = 0u;
                    } else {
                        if (_e3127_ == 0u) {
                            phi_7393_ = (_e3126_ & 2147483648u);
                        } else {
                            phi_7393_ = _e3126_;
                        }
                        let _e3132_ = phi_7393_;
                        phi_7394_ = _e3132_;
                    }
                    let _e3134_ = phi_7394_;
                    let _e3136_ = bitcast<u32>(_e3111_[2]);
                    let _e3137_ = (_e3136_ & 2139095040u);
                    if (_e3137_ == 2139095040u) {
                        phi_7409_ = 0u;
                    } else {
                        if (_e3137_ == 0u) {
                            phi_7408_ = (_e3136_ & 2147483648u);
                        } else {
                            phi_7408_ = _e3136_;
                        }
                        let _e3142_ = phi_7408_;
                        phi_7409_ = _e3142_;
                    }
                    let _e3144_ = phi_7409_;
                    let _e3146_ = bitcast<u32>(_e3111_[3]);
                    let _e3147_ = (_e3146_ & 2139095040u);
                    if (_e3147_ == 2139095040u) {
                        phi_7424_ = 0u;
                    } else {
                        if (_e3147_ == 0u) {
                            phi_7423_ = (_e3146_ & 2147483648u);
                        } else {
                            phi_7423_ = _e3146_;
                        }
                        let _e3152_ = phi_7423_;
                        phi_7424_ = _e3152_;
                    }
                    let _e3154_ = phi_7424_;
                    let _e3157_ = local_5_;
                    let _e3162_ = bitcast<u32>(_e3157_[0]);
                    let _e3163_ = (_e3162_ & 2139095040u);
                    if (_e3163_ == 2139095040u) {
                        phi_7447_ = 0u;
                    } else {
                        if (_e3163_ == 0u) {
                            phi_7446_ = (_e3162_ & 2147483648u);
                        } else {
                            phi_7446_ = _e3162_;
                        }
                        let _e3168_ = phi_7446_;
                        phi_7447_ = _e3168_;
                    }
                    let _e3170_ = phi_7447_;
                    let _e3172_ = bitcast<u32>(_e3157_[1]);
                    let _e3173_ = (_e3172_ & 2139095040u);
                    if (_e3173_ == 2139095040u) {
                        phi_7462_ = 0u;
                    } else {
                        if (_e3173_ == 0u) {
                            phi_7461_ = (_e3172_ & 2147483648u);
                        } else {
                            phi_7461_ = _e3172_;
                        }
                        let _e3178_ = phi_7461_;
                        phi_7462_ = _e3178_;
                    }
                    let _e3180_ = phi_7462_;
                    let _e3182_ = bitcast<u32>(_e3157_[2]);
                    let _e3183_ = (_e3182_ & 2139095040u);
                    if (_e3183_ == 2139095040u) {
                        phi_7477_ = 0u;
                    } else {
                        if (_e3183_ == 0u) {
                            phi_7476_ = (_e3182_ & 2147483648u);
                        } else {
                            phi_7476_ = _e3182_;
                        }
                        let _e3188_ = phi_7476_;
                        phi_7477_ = _e3188_;
                    }
                    let _e3190_ = phi_7477_;
                    let _e3192_ = bitcast<u32>(_e3157_[3]);
                    let _e3193_ = (_e3192_ & 2139095040u);
                    if (_e3193_ == 2139095040u) {
                        phi_7492_ = 0u;
                    } else {
                        if (_e3193_ == 0u) {
                            phi_7491_ = (_e3192_ & 2147483648u);
                        } else {
                            phi_7491_ = _e3192_;
                        }
                        let _e3198_ = phi_7491_;
                        phi_7492_ = _e3198_;
                    }
                    let _e3200_ = phi_7492_;
                    let _e3210_ = type_32_(_e3110_.member, type_27_(type_24_(array<f32, 4>(bitcast<f32>(_e3124_), bitcast<f32>(_e3134_), bitcast<f32>(_e3144_), bitcast<f32>(_e3154_))), type_24_(array<f32, 4>(bitcast<f32>(_e3170_), bitcast<f32>(_e3180_), bitcast<f32>(_e3190_), bitcast<f32>(_e3200_)))), _e3110_.member_2_, _e3110_.member_3_);
                    if _e1304_ {
                    } else {
                        phi_18515_ = true;
                        break;
                    }
                    let _e3211_ = lane_buf[_e120_];
                    if _e1307_ {
                    } else {
                        phi_18515_ = true;
                        break;
                    }
                    let _e3216_ = lane_buf[_e1306_];
                    let _e3222_ = local_24_;
                    if _e949_ {
                        phi_7563_ = _e3222_.member_3_.member_1_;
                        phi_7564_ = _e3222_.member_3_.member;
                    } else {
                        let _e3227_ = ((_e120_ - 1u) & 255u);
                        if (_e3227_ < 512u) {
                        } else {
                            phi_18515_ = true;
                            break;
                        }
                        let _e3230_ = lane_buf[_e3227_];
                        let _e3237_ = (256u + _e3227_);
                        if (_e3237_ < 512u) {
                        } else {
                            phi_18515_ = true;
                            break;
                        }
                        let _e3240_ = lane_buf[_e3237_];
                        phi_7563_ = type_24_(array<f32, 4>(_e3240_.x, _e3240_.y, _e3240_.z, _e3240_.w));
                        phi_7564_ = type_24_(array<f32, 4>(_e3230_.x, _e3230_.y, _e3230_.z, _e3230_.w));
                    }
                    let _e3248_ = phi_7563_;
                    let _e3250_ = phi_7564_;
                    let _e3272_ = select(0f, _e3250_.member[0], (bitcast<f32>((bitcast<u32>(_e3250_.member[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3274_ = select(0f, _e3250_.member[1], (bitcast<f32>((bitcast<u32>(_e3250_.member[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3276_ = select(0f, _e3250_.member[2], (bitcast<f32>((bitcast<u32>(_e3250_.member[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3278_ = select(0f, _e3250_.member[3], (bitcast<f32>((bitcast<u32>(_e3250_.member[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3279_ = (_e3272_ != _e3272_);
                    if _e3279_ {
                        phi_13739_ = true;
                    } else {
                        phi_13739_ = (-1000000000000000000f >= _e3272_);
                    }
                    let _e3282_ = phi_13739_;
                    let _e3283_ = select(_e3272_, -1000000000000000000f, _e3282_);
                    let _e3284_ = (_e3274_ != _e3274_);
                    if _e3284_ {
                        phi_13754_ = true;
                    } else {
                        phi_13754_ = (-1000000000000000000f >= _e3274_);
                    }
                    let _e3287_ = phi_13754_;
                    let _e3288_ = select(_e3274_, -1000000000000000000f, _e3287_);
                    let _e3289_ = (_e3276_ != _e3276_);
                    if _e3289_ {
                        phi_13769_ = true;
                    } else {
                        phi_13769_ = (-1000000000000000000f >= _e3276_);
                    }
                    let _e3292_ = phi_13769_;
                    let _e3293_ = select(_e3276_, -1000000000000000000f, _e3292_);
                    let _e3294_ = (_e3278_ != _e3278_);
                    if _e3294_ {
                        phi_13784_ = true;
                    } else {
                        phi_13784_ = (-1000000000000000000f >= _e3278_);
                    }
                    let _e3297_ = phi_13784_;
                    let _e3298_ = select(_e3278_, -1000000000000000000f, _e3297_);
                    if (_e3283_ != _e3283_) {
                        phi_13799_ = true;
                    } else {
                        phi_13799_ = (1000000000000000000f <= _e3283_);
                    }
                    let _e3302_ = phi_13799_;
                    let _e3303_ = select(_e3283_, 1000000000000000000f, _e3302_);
                    if (_e3288_ != _e3288_) {
                        phi_13814_ = true;
                    } else {
                        phi_13814_ = (1000000000000000000f <= _e3288_);
                    }
                    let _e3307_ = phi_13814_;
                    let _e3308_ = select(_e3288_, 1000000000000000000f, _e3307_);
                    if (_e3293_ != _e3293_) {
                        phi_13829_ = true;
                    } else {
                        phi_13829_ = (1000000000000000000f <= _e3293_);
                    }
                    let _e3312_ = phi_13829_;
                    let _e3313_ = select(_e3293_, 1000000000000000000f, _e3312_);
                    if (_e3298_ != _e3298_) {
                        phi_13844_ = true;
                    } else {
                        phi_13844_ = (1000000000000000000f <= _e3298_);
                    }
                    let _e3317_ = phi_13844_;
                    let _e3318_ = select(_e3298_, 1000000000000000000f, _e3317_);
                    let _e3332_ = select(0f, _e3211_.x, (bitcast<f32>((bitcast<u32>(_e3211_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3334_ = select(0f, _e3211_.y, (bitcast<f32>((bitcast<u32>(_e3211_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3336_ = select(0f, _e3211_.z, (bitcast<f32>((bitcast<u32>(_e3211_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3338_ = select(0f, _e3211_.w, (bitcast<f32>((bitcast<u32>(_e3211_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3339_ = (_e3332_ != _e3332_);
                    if _e3339_ {
                        phi_13859_ = true;
                    } else {
                        phi_13859_ = (-1000000000000000000f >= _e3332_);
                    }
                    let _e3342_ = phi_13859_;
                    let _e3343_ = select(_e3332_, -1000000000000000000f, _e3342_);
                    let _e3344_ = (_e3334_ != _e3334_);
                    if _e3344_ {
                        phi_13874_ = true;
                    } else {
                        phi_13874_ = (-1000000000000000000f >= _e3334_);
                    }
                    let _e3347_ = phi_13874_;
                    let _e3348_ = select(_e3334_, -1000000000000000000f, _e3347_);
                    let _e3349_ = (_e3336_ != _e3336_);
                    if _e3349_ {
                        phi_13889_ = true;
                    } else {
                        phi_13889_ = (-1000000000000000000f >= _e3336_);
                    }
                    let _e3352_ = phi_13889_;
                    let _e3353_ = select(_e3336_, -1000000000000000000f, _e3352_);
                    let _e3354_ = (_e3338_ != _e3338_);
                    if _e3354_ {
                        phi_13904_ = true;
                    } else {
                        phi_13904_ = (-1000000000000000000f >= _e3338_);
                    }
                    let _e3357_ = phi_13904_;
                    let _e3358_ = select(_e3338_, -1000000000000000000f, _e3357_);
                    if (_e3343_ != _e3343_) {
                        phi_13919_ = true;
                    } else {
                        phi_13919_ = (1000000000000000000f <= _e3343_);
                    }
                    let _e3362_ = phi_13919_;
                    let _e3363_ = select(_e3343_, 1000000000000000000f, _e3362_);
                    if (_e3348_ != _e3348_) {
                        phi_13934_ = true;
                    } else {
                        phi_13934_ = (1000000000000000000f <= _e3348_);
                    }
                    let _e3367_ = phi_13934_;
                    let _e3368_ = select(_e3348_, 1000000000000000000f, _e3367_);
                    if (_e3353_ != _e3353_) {
                        phi_13949_ = true;
                    } else {
                        phi_13949_ = (1000000000000000000f <= _e3353_);
                    }
                    let _e3372_ = phi_13949_;
                    let _e3373_ = select(_e3353_, 1000000000000000000f, _e3372_);
                    if (_e3358_ != _e3358_) {
                        phi_13964_ = true;
                    } else {
                        phi_13964_ = (1000000000000000000f <= _e3358_);
                    }
                    let _e3377_ = phi_13964_;
                    let _e3378_ = select(_e3358_, 1000000000000000000f, _e3377_);
                    let _e3379_ = array<f32, 4>(_e3363_, _e3368_, _e3373_, _e3378_);
                    let _e3380_ = (_e3303_ != _e3303_);
                    if _e3380_ {
                        phi_13979_ = true;
                    } else {
                        phi_13979_ = (_e3363_ <= _e3303_);
                    }
                    let _e3383_ = phi_13979_;
                    let _e3384_ = select(_e3303_, _e3363_, _e3383_);
                    let _e3385_ = (_e3308_ != _e3308_);
                    if _e3385_ {
                        phi_13994_ = true;
                    } else {
                        phi_13994_ = (_e3368_ <= _e3308_);
                    }
                    let _e3388_ = phi_13994_;
                    let _e3389_ = select(_e3308_, _e3368_, _e3388_);
                    let _e3390_ = (_e3313_ != _e3313_);
                    if _e3390_ {
                        phi_14009_ = true;
                    } else {
                        phi_14009_ = (_e3373_ <= _e3313_);
                    }
                    let _e3393_ = phi_14009_;
                    let _e3394_ = select(_e3313_, _e3373_, _e3393_);
                    let _e3395_ = (_e3318_ != _e3318_);
                    if _e3395_ {
                        phi_14024_ = true;
                    } else {
                        phi_14024_ = (_e3378_ <= _e3318_);
                    }
                    let _e3398_ = phi_14024_;
                    let _e3399_ = select(_e3318_, _e3378_, _e3398_);
                    if _e3380_ {
                        phi_14039_ = true;
                    } else {
                        phi_14039_ = (_e3363_ >= _e3303_);
                    }
                    let _e3402_ = phi_14039_;
                    let _e3403_ = select(_e3303_, _e3363_, _e3402_);
                    if _e3385_ {
                        phi_14054_ = true;
                    } else {
                        phi_14054_ = (_e3368_ >= _e3308_);
                    }
                    let _e3406_ = phi_14054_;
                    let _e3407_ = select(_e3308_, _e3368_, _e3406_);
                    if _e3390_ {
                        phi_14069_ = true;
                    } else {
                        phi_14069_ = (_e3373_ >= _e3313_);
                    }
                    let _e3410_ = phi_14069_;
                    let _e3411_ = select(_e3313_, _e3373_, _e3410_);
                    if _e3395_ {
                        phi_14084_ = true;
                    } else {
                        phi_14084_ = (_e3378_ >= _e3318_);
                    }
                    let _e3414_ = phi_14084_;
                    let _e3415_ = select(_e3318_, _e3378_, _e3414_);
                    let _e3416_ = (_e3384_ != _e3384_);
                    if _e3416_ {
                        phi_14099_ = true;
                    } else {
                        phi_14099_ = (-1f >= _e3384_);
                    }
                    let _e3419_ = phi_14099_;
                    let _e3420_ = select(_e3384_, -1f, _e3419_);
                    let _e3421_ = (_e3389_ != _e3389_);
                    if _e3421_ {
                        phi_14114_ = true;
                    } else {
                        phi_14114_ = (-1f >= _e3389_);
                    }
                    let _e3424_ = phi_14114_;
                    let _e3425_ = select(_e3389_, -1f, _e3424_);
                    let _e3426_ = (_e3394_ != _e3394_);
                    if _e3426_ {
                        phi_14129_ = true;
                    } else {
                        phi_14129_ = (-1f >= _e3394_);
                    }
                    let _e3429_ = phi_14129_;
                    let _e3430_ = select(_e3394_, -1f, _e3429_);
                    let _e3431_ = (_e3399_ != _e3399_);
                    if _e3431_ {
                        phi_14144_ = true;
                    } else {
                        phi_14144_ = (-1f >= _e3399_);
                    }
                    let _e3434_ = phi_14144_;
                    let _e3435_ = select(_e3399_, -1f, _e3434_);
                    if (_e3420_ != _e3420_) {
                        phi_14159_ = true;
                    } else {
                        phi_14159_ = (1f <= _e3420_);
                    }
                    let _e3439_ = phi_14159_;
                    let _e3440_ = select(_e3420_, 1f, _e3439_);
                    if (_e3425_ != _e3425_) {
                        phi_14174_ = true;
                    } else {
                        phi_14174_ = (1f <= _e3425_);
                    }
                    let _e3444_ = phi_14174_;
                    let _e3445_ = select(_e3425_, 1f, _e3444_);
                    if (_e3430_ != _e3430_) {
                        phi_14189_ = true;
                    } else {
                        phi_14189_ = (1f <= _e3430_);
                    }
                    let _e3449_ = phi_14189_;
                    let _e3450_ = select(_e3430_, 1f, _e3449_);
                    if (_e3435_ != _e3435_) {
                        phi_14204_ = true;
                    } else {
                        phi_14204_ = (1f <= _e3435_);
                    }
                    let _e3454_ = phi_14204_;
                    let _e3455_ = select(_e3435_, 1f, _e3454_);
                    let _e3456_ = (_e3403_ != _e3403_);
                    if _e3456_ {
                        phi_14219_ = true;
                    } else {
                        phi_14219_ = (-1f >= _e3403_);
                    }
                    let _e3459_ = phi_14219_;
                    let _e3460_ = select(_e3403_, -1f, _e3459_);
                    let _e3461_ = (_e3407_ != _e3407_);
                    if _e3461_ {
                        phi_14234_ = true;
                    } else {
                        phi_14234_ = (-1f >= _e3407_);
                    }
                    let _e3464_ = phi_14234_;
                    let _e3465_ = select(_e3407_, -1f, _e3464_);
                    let _e3466_ = (_e3411_ != _e3411_);
                    if _e3466_ {
                        phi_14249_ = true;
                    } else {
                        phi_14249_ = (-1f >= _e3411_);
                    }
                    let _e3469_ = phi_14249_;
                    let _e3470_ = select(_e3411_, -1f, _e3469_);
                    let _e3471_ = (_e3415_ != _e3415_);
                    if _e3471_ {
                        phi_14264_ = true;
                    } else {
                        phi_14264_ = (-1f >= _e3415_);
                    }
                    let _e3474_ = phi_14264_;
                    let _e3475_ = select(_e3415_, -1f, _e3474_);
                    if (_e3460_ != _e3460_) {
                        phi_14279_ = true;
                    } else {
                        phi_14279_ = (1f <= _e3460_);
                    }
                    let _e3479_ = phi_14279_;
                    let _e3480_ = select(_e3460_, 1f, _e3479_);
                    if (_e3465_ != _e3465_) {
                        phi_14294_ = true;
                    } else {
                        phi_14294_ = (1f <= _e3465_);
                    }
                    let _e3484_ = phi_14294_;
                    let _e3485_ = select(_e3465_, 1f, _e3484_);
                    if (_e3470_ != _e3470_) {
                        phi_14309_ = true;
                    } else {
                        phi_14309_ = (1f <= _e3470_);
                    }
                    let _e3489_ = phi_14309_;
                    let _e3490_ = select(_e3470_, 1f, _e3489_);
                    if (_e3475_ != _e3475_) {
                        phi_14324_ = true;
                    } else {
                        phi_14324_ = (1f <= _e3475_);
                    }
                    let _e3494_ = phi_14324_;
                    let _e3495_ = select(_e3475_, 1f, _e3494_);
                    let _e3500_ = (0.5f * (_e3480_ + _e3440_));
                    let _e3501_ = (0.5f * (_e3485_ + _e3445_));
                    let _e3502_ = (0.5f * (_e3490_ + _e3450_));
                    let _e3503_ = (0.5f * (_e3495_ + _e3455_));
                    let _e3504_ = (_e3480_ - _e3440_);
                    let _e3505_ = (_e3485_ - _e3445_);
                    let _e3506_ = (_e3490_ - _e3450_);
                    let _e3507_ = (_e3495_ - _e3455_);
                    let _e3508_ = (0.5f * _e3504_);
                    let _e3509_ = (0.5f * _e3505_);
                    let _e3510_ = (0.5f * _e3506_);
                    let _e3511_ = (0.5f * _e3507_);
                    let _e3512_ = (_e3500_ * _e3500_);
                    let _e3513_ = (_e3501_ * _e3501_);
                    let _e3514_ = (_e3502_ * _e3502_);
                    let _e3515_ = (_e3503_ * _e3503_);
                    let _e3516_ = (_e3508_ * _e3508_);
                    let _e3517_ = (_e3509_ * _e3509_);
                    let _e3518_ = (_e3510_ * _e3510_);
                    let _e3519_ = (_e3511_ * _e3511_);
                    if (_e3504_ != _e3504_) {
                        phi_14339_ = true;
                    } else {
                        phi_14339_ = (0f >= _e3504_);
                    }
                    let _e3523_ = phi_14339_;
                    let _e3524_ = select(_e3504_, 0f, _e3523_);
                    if (_e3505_ != _e3505_) {
                        phi_14354_ = true;
                    } else {
                        phi_14354_ = (0f >= _e3505_);
                    }
                    let _e3528_ = phi_14354_;
                    let _e3529_ = select(_e3505_, 0f, _e3528_);
                    if (_e3506_ != _e3506_) {
                        phi_14369_ = true;
                    } else {
                        phi_14369_ = (0f >= _e3506_);
                    }
                    let _e3533_ = phi_14369_;
                    let _e3534_ = select(_e3506_, 0f, _e3533_);
                    if (_e3507_ != _e3507_) {
                        phi_14384_ = true;
                    } else {
                        phi_14384_ = (0f >= _e3507_);
                    }
                    let _e3538_ = phi_14384_;
                    let _e3539_ = select(_e3507_, 0f, _e3538_);
                    let _e3540_ = (_e3403_ - _e3384_);
                    let _e3541_ = (_e3407_ - _e3389_);
                    let _e3542_ = (_e3411_ - _e3394_);
                    let _e3543_ = (_e3415_ - _e3399_);
                    if _e3456_ {
                        phi_14399_ = true;
                    } else {
                        phi_14399_ = (-1f <= _e3403_);
                    }
                    let _e3546_ = phi_14399_;
                    if _e3461_ {
                        phi_14414_ = true;
                    } else {
                        phi_14414_ = (-1f <= _e3407_);
                    }
                    let _e3550_ = phi_14414_;
                    if _e3466_ {
                        phi_14429_ = true;
                    } else {
                        phi_14429_ = (-1f <= _e3411_);
                    }
                    let _e3554_ = phi_14429_;
                    if _e3471_ {
                        phi_14444_ = true;
                    } else {
                        phi_14444_ = (-1f <= _e3415_);
                    }
                    let _e3558_ = phi_14444_;
                    let _e3560_ = (select(_e3403_, -1f, _e3546_) - _e3384_);
                    let _e3561_ = (select(_e3407_, -1f, _e3550_) - _e3389_);
                    let _e3562_ = (select(_e3411_, -1f, _e3554_) - _e3394_);
                    let _e3563_ = (select(_e3415_, -1f, _e3558_) - _e3399_);
                    if (_e3560_ != _e3560_) {
                        phi_14459_ = true;
                    } else {
                        phi_14459_ = (0f >= _e3560_);
                    }
                    let _e3567_ = phi_14459_;
                    if (_e3561_ != _e3561_) {
                        phi_14474_ = true;
                    } else {
                        phi_14474_ = (0f >= _e3561_);
                    }
                    let _e3572_ = phi_14474_;
                    if (_e3562_ != _e3562_) {
                        phi_14489_ = true;
                    } else {
                        phi_14489_ = (0f >= _e3562_);
                    }
                    let _e3577_ = phi_14489_;
                    if (_e3563_ != _e3563_) {
                        phi_14504_ = true;
                    } else {
                        phi_14504_ = (0f >= _e3563_);
                    }
                    let _e3582_ = phi_14504_;
                    if _e3416_ {
                        phi_14519_ = true;
                    } else {
                        phi_14519_ = (1f >= _e3384_);
                    }
                    let _e3586_ = phi_14519_;
                    if _e3421_ {
                        phi_14534_ = true;
                    } else {
                        phi_14534_ = (1f >= _e3389_);
                    }
                    let _e3590_ = phi_14534_;
                    if _e3426_ {
                        phi_14549_ = true;
                    } else {
                        phi_14549_ = (1f >= _e3394_);
                    }
                    let _e3594_ = phi_14549_;
                    if _e3431_ {
                        phi_14564_ = true;
                    } else {
                        phi_14564_ = (1f >= _e3399_);
                    }
                    let _e3598_ = phi_14564_;
                    let _e3600_ = (_e3403_ - select(_e3384_, 1f, _e3586_));
                    let _e3601_ = (_e3407_ - select(_e3389_, 1f, _e3590_));
                    let _e3602_ = (_e3411_ - select(_e3394_, 1f, _e3594_));
                    let _e3603_ = (_e3415_ - select(_e3399_, 1f, _e3598_));
                    if (_e3600_ != _e3600_) {
                        phi_14579_ = true;
                    } else {
                        phi_14579_ = (0f >= _e3600_);
                    }
                    let _e3607_ = phi_14579_;
                    if (_e3601_ != _e3601_) {
                        phi_14594_ = true;
                    } else {
                        phi_14594_ = (0f >= _e3601_);
                    }
                    let _e3612_ = phi_14594_;
                    if (_e3602_ != _e3602_) {
                        phi_14609_ = true;
                    } else {
                        phi_14609_ = (0f >= _e3602_);
                    }
                    let _e3617_ = phi_14609_;
                    if (_e3603_ != _e3603_) {
                        phi_14624_ = true;
                    } else {
                        phi_14624_ = (0f >= _e3603_);
                    }
                    let _e3622_ = phi_14624_;
                    let _e3626_ = bitcast<f32>((bitcast<u32>(_e3303_) & 2147483647u));
                    let _e3629_ = bitcast<f32>((bitcast<u32>(_e3308_) & 2147483647u));
                    let _e3632_ = bitcast<f32>((bitcast<u32>(_e3313_) & 2147483647u));
                    let _e3635_ = bitcast<f32>((bitcast<u32>(_e3318_) & 2147483647u));
                    let _e3638_ = bitcast<f32>((bitcast<u32>(_e3363_) & 2147483647u));
                    let _e3641_ = bitcast<f32>((bitcast<u32>(_e3368_) & 2147483647u));
                    let _e3644_ = bitcast<f32>((bitcast<u32>(_e3373_) & 2147483647u));
                    let _e3647_ = bitcast<f32>((bitcast<u32>(_e3378_) & 2147483647u));
                    if (_e3626_ != _e3626_) {
                        phi_14639_ = true;
                    } else {
                        phi_14639_ = (_e3638_ >= _e3626_);
                    }
                    let _e3651_ = phi_14639_;
                    let _e3652_ = select(_e3626_, _e3638_, _e3651_);
                    if (_e3629_ != _e3629_) {
                        phi_14654_ = true;
                    } else {
                        phi_14654_ = (_e3641_ >= _e3629_);
                    }
                    let _e3656_ = phi_14654_;
                    let _e3657_ = select(_e3629_, _e3641_, _e3656_);
                    if (_e3632_ != _e3632_) {
                        phi_14669_ = true;
                    } else {
                        phi_14669_ = (_e3644_ >= _e3632_);
                    }
                    let _e3661_ = phi_14669_;
                    let _e3662_ = select(_e3632_, _e3644_, _e3661_);
                    if (_e3635_ != _e3635_) {
                        phi_14684_ = true;
                    } else {
                        phi_14684_ = (_e3647_ >= _e3635_);
                    }
                    let _e3666_ = phi_14684_;
                    let _e3667_ = select(_e3635_, _e3647_, _e3666_);
                    let _e3684_ = (_e3652_ <= 0f);
                    let _e3685_ = select((_e3500_ * (1.5f - (0.5f * (_e3512_ + _e3516_)))), 0f, _e3684_);
                    let _e3686_ = (_e3657_ <= 0f);
                    let _e3687_ = select((_e3501_ * (1.5f - (0.5f * (_e3513_ + _e3517_)))), 0f, _e3686_);
                    let _e3688_ = (_e3662_ <= 0f);
                    let _e3689_ = select((_e3502_ * (1.5f - (0.5f * (_e3514_ + _e3518_)))), 0f, _e3688_);
                    let _e3690_ = (_e3667_ <= 0f);
                    let _e3691_ = select((_e3503_ * (1.5f - (0.5f * (_e3515_ + _e3519_)))), 0f, _e3690_);
                    let _e3733_ = select(((_e3512_ + (0.33333334f * _e3516_)) - (((_e3512_ * _e3512_) + ((2f * _e3512_) * _e3516_)) + (0.2f * (_e3516_ * _e3516_)))), 0f, _e3684_);
                    let _e3734_ = select(((_e3513_ + (0.33333334f * _e3517_)) - (((_e3513_ * _e3513_) + ((2f * _e3513_) * _e3517_)) + (0.2f * (_e3517_ * _e3517_)))), 0f, _e3686_);
                    let _e3735_ = select(((_e3514_ + (0.33333334f * _e3518_)) - (((_e3514_ * _e3514_) + ((2f * _e3514_) * _e3518_)) + (0.2f * (_e3518_ * _e3518_)))), 0f, _e3688_);
                    let _e3736_ = select(((_e3515_ + (0.33333334f * _e3519_)) - (((_e3515_ * _e3515_) + ((2f * _e3515_) * _e3519_)) + (0.2f * (_e3519_ * _e3519_)))), 0f, _e3690_);
                    if (_e3652_ <= 1f) {
                        if (_e3657_ <= 1f) {
                            let _e3740_ = (_e3662_ <= 1f);
                            if _e3740_ {
                                phi_7999_ = (_e3667_ <= 1f);
                            } else {
                                phi_7999_ = bool();
                            }
                            let _e3743_ = phi_7999_;
                            phi_8001_ = _e3743_;
                            phi_8002_ = select(true, false, _e3740_);
                        } else {
                            phi_8001_ = bool();
                            phi_8002_ = true;
                        }
                        let _e3746_ = phi_8001_;
                        let _e3748_ = phi_8002_;
                        phi_8003_ = _e3746_;
                        phi_8004_ = _e3748_;
                    } else {
                        phi_8003_ = bool();
                        phi_8004_ = true;
                    }
                    let _e3750_ = phi_8003_;
                    let _e3752_ = phi_8004_;
                    if select(_e3750_, false, _e3752_) {
                        phi_8382_ = type_33_(type_24_(array<f32, 4>(_e3685_, _e3687_, _e3689_, _e3691_)), type_24_(array<f32, 4>(_e3733_, _e3734_, _e3735_, _e3736_)), type_24_(_e3379_));
                    } else {
                        if (_e3540_ != _e3540_) {
                            phi_14699_ = true;
                        } else {
                            phi_14699_ = (0.000001f >= _e3540_);
                        }
                        let _e3757_ = phi_14699_;
                        let _e3758_ = select(_e3540_, 0.000001f, _e3757_);
                        if (_e3541_ != _e3541_) {
                            phi_14714_ = true;
                        } else {
                            phi_14714_ = (0.000001f >= _e3541_);
                        }
                        let _e3762_ = phi_14714_;
                        let _e3763_ = select(_e3541_, 0.000001f, _e3762_);
                        if (_e3542_ != _e3542_) {
                            phi_14729_ = true;
                        } else {
                            phi_14729_ = (0.000001f >= _e3542_);
                        }
                        let _e3767_ = phi_14729_;
                        let _e3768_ = select(_e3542_, 0.000001f, _e3767_);
                        if (_e3543_ != _e3543_) {
                            phi_14744_ = true;
                        } else {
                            phi_14744_ = (0.000001f >= _e3543_);
                        }
                        let _e3772_ = phi_14744_;
                        let _e3773_ = select(_e3543_, 0.000001f, _e3772_);
                        let _e3776_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3758_)));
                        let _e3779_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3763_)));
                        let _e3782_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3768_)));
                        let _e3785_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3773_)));
                        let _e3795_ = (_e3776_ + (_e3776_ * (1f - (_e3758_ * _e3776_))));
                        let _e3797_ = (_e3779_ + (_e3779_ * (1f - (_e3763_ * _e3779_))));
                        let _e3799_ = (_e3782_ + (_e3782_ * (1f - (_e3768_ * _e3782_))));
                        let _e3801_ = (_e3785_ + (_e3785_ * (1f - (_e3773_ * _e3785_))));
                        let _e3811_ = (_e3795_ + (_e3795_ * (1f - (_e3758_ * _e3795_))));
                        let _e3813_ = (_e3797_ + (_e3797_ * (1f - (_e3763_ * _e3797_))));
                        let _e3815_ = (_e3799_ + (_e3799_ * (1f - (_e3768_ * _e3799_))));
                        let _e3817_ = (_e3801_ + (_e3801_ * (1f - (_e3773_ * _e3801_))));
                        let _e3870_ = (select((select(0f, 1f, (-1f <= _e3384_)) * select(0f, 1f, (_e3403_ <= 1f))), 1f, (_e3540_ <= 0.000001f)) <= 0f);
                        let _e3872_ = (select((select(0f, 1f, (-1f <= _e3389_)) * select(0f, 1f, (_e3407_ <= 1f))), 1f, (_e3541_ <= 0.000001f)) <= 0f);
                        let _e3874_ = (select((select(0f, 1f, (-1f <= _e3394_)) * select(0f, 1f, (_e3411_ <= 1f))), 1f, (_e3542_ <= 0.000001f)) <= 0f);
                        let _e3876_ = (select((select(0f, 1f, (-1f <= _e3399_)) * select(0f, 1f, (_e3415_ <= 1f))), 1f, (_e3543_ <= 0.000001f)) <= 0f);
                        let _e3882_ = (1f <= _e3384_);
                        let _e3884_ = (1f <= _e3389_);
                        let _e3886_ = (1f <= _e3394_);
                        let _e3888_ = (1f <= _e3399_);
                        let _e3890_ = (_e3403_ <= -1f);
                        let _e3892_ = (_e3407_ <= -1f);
                        let _e3894_ = (_e3411_ <= -1f);
                        let _e3896_ = (_e3415_ <= -1f);
                        phi_8382_ = type_33_(type_24_(array<f32, 4>(select(select(select(_e3685_, (((select(_e3600_, 0f, _e3607_) - select(_e3560_, 0f, _e3567_)) + (_e3524_ * _e3685_)) * _e3811_), _e3870_), 1f, _e3882_), -1f, _e3890_), select(select(select(_e3687_, (((select(_e3601_, 0f, _e3612_) - select(_e3561_, 0f, _e3572_)) + (_e3529_ * _e3687_)) * _e3813_), _e3872_), 1f, _e3884_), -1f, _e3892_), select(select(select(_e3689_, (((select(_e3602_, 0f, _e3617_) - select(_e3562_, 0f, _e3577_)) + (_e3534_ * _e3689_)) * _e3815_), _e3874_), 1f, _e3886_), -1f, _e3894_), select(select(select(_e3691_, (((select(_e3603_, 0f, _e3622_) - select(_e3563_, 0f, _e3582_)) + (_e3539_ * _e3691_)) * _e3817_), _e3876_), 1f, _e3888_), -1f, _e3896_))), type_24_(array<f32, 4>(select(select(select(_e3733_, ((_e3524_ * _e3733_) * _e3811_), _e3870_), 0f, _e3882_), 0f, _e3890_), select(select(select(_e3734_, ((_e3529_ * _e3734_) * _e3813_), _e3872_), 0f, _e3884_), 0f, _e3892_), select(select(select(_e3735_, ((_e3534_ * _e3735_) * _e3815_), _e3874_), 0f, _e3886_), 0f, _e3894_), select(select(select(_e3736_, ((_e3539_ * _e3736_) * _e3817_), _e3876_), 0f, _e3888_), 0f, _e3896_))), type_24_(_e3379_));
                    }
                    let _e3917_ = phi_8382_;
                    let _e3940_ = select(0f, _e3248_.member[0], (bitcast<f32>((bitcast<u32>(_e3248_.member[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3942_ = select(0f, _e3248_.member[1], (bitcast<f32>((bitcast<u32>(_e3248_.member[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3944_ = select(0f, _e3248_.member[2], (bitcast<f32>((bitcast<u32>(_e3248_.member[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3946_ = select(0f, _e3248_.member[3], (bitcast<f32>((bitcast<u32>(_e3248_.member[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3947_ = (_e3940_ != _e3940_);
                    if _e3947_ {
                        phi_14759_ = true;
                    } else {
                        phi_14759_ = (-1000000000000000000f >= _e3940_);
                    }
                    let _e3950_ = phi_14759_;
                    let _e3951_ = select(_e3940_, -1000000000000000000f, _e3950_);
                    let _e3952_ = (_e3942_ != _e3942_);
                    if _e3952_ {
                        phi_14774_ = true;
                    } else {
                        phi_14774_ = (-1000000000000000000f >= _e3942_);
                    }
                    let _e3955_ = phi_14774_;
                    let _e3956_ = select(_e3942_, -1000000000000000000f, _e3955_);
                    let _e3957_ = (_e3944_ != _e3944_);
                    if _e3957_ {
                        phi_14789_ = true;
                    } else {
                        phi_14789_ = (-1000000000000000000f >= _e3944_);
                    }
                    let _e3960_ = phi_14789_;
                    let _e3961_ = select(_e3944_, -1000000000000000000f, _e3960_);
                    let _e3962_ = (_e3946_ != _e3946_);
                    if _e3962_ {
                        phi_14804_ = true;
                    } else {
                        phi_14804_ = (-1000000000000000000f >= _e3946_);
                    }
                    let _e3965_ = phi_14804_;
                    let _e3966_ = select(_e3946_, -1000000000000000000f, _e3965_);
                    if (_e3951_ != _e3951_) {
                        phi_14819_ = true;
                    } else {
                        phi_14819_ = (1000000000000000000f <= _e3951_);
                    }
                    let _e3970_ = phi_14819_;
                    let _e3971_ = select(_e3951_, 1000000000000000000f, _e3970_);
                    if (_e3956_ != _e3956_) {
                        phi_14834_ = true;
                    } else {
                        phi_14834_ = (1000000000000000000f <= _e3956_);
                    }
                    let _e3975_ = phi_14834_;
                    let _e3976_ = select(_e3956_, 1000000000000000000f, _e3975_);
                    if (_e3961_ != _e3961_) {
                        phi_14849_ = true;
                    } else {
                        phi_14849_ = (1000000000000000000f <= _e3961_);
                    }
                    let _e3980_ = phi_14849_;
                    let _e3981_ = select(_e3961_, 1000000000000000000f, _e3980_);
                    if (_e3966_ != _e3966_) {
                        phi_14864_ = true;
                    } else {
                        phi_14864_ = (1000000000000000000f <= _e3966_);
                    }
                    let _e3985_ = phi_14864_;
                    let _e3986_ = select(_e3966_, 1000000000000000000f, _e3985_);
                    let _e4000_ = select(0f, _e3216_.x, (bitcast<f32>((bitcast<u32>(_e3216_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e4002_ = select(0f, _e3216_.y, (bitcast<f32>((bitcast<u32>(_e3216_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e4004_ = select(0f, _e3216_.z, (bitcast<f32>((bitcast<u32>(_e3216_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e4006_ = select(0f, _e3216_.w, (bitcast<f32>((bitcast<u32>(_e3216_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e4007_ = (_e4000_ != _e4000_);
                    if _e4007_ {
                        phi_14879_ = true;
                    } else {
                        phi_14879_ = (-1000000000000000000f >= _e4000_);
                    }
                    let _e4010_ = phi_14879_;
                    let _e4011_ = select(_e4000_, -1000000000000000000f, _e4010_);
                    let _e4012_ = (_e4002_ != _e4002_);
                    if _e4012_ {
                        phi_14894_ = true;
                    } else {
                        phi_14894_ = (-1000000000000000000f >= _e4002_);
                    }
                    let _e4015_ = phi_14894_;
                    let _e4016_ = select(_e4002_, -1000000000000000000f, _e4015_);
                    let _e4017_ = (_e4004_ != _e4004_);
                    if _e4017_ {
                        phi_14909_ = true;
                    } else {
                        phi_14909_ = (-1000000000000000000f >= _e4004_);
                    }
                    let _e4020_ = phi_14909_;
                    let _e4021_ = select(_e4004_, -1000000000000000000f, _e4020_);
                    let _e4022_ = (_e4006_ != _e4006_);
                    if _e4022_ {
                        phi_14924_ = true;
                    } else {
                        phi_14924_ = (-1000000000000000000f >= _e4006_);
                    }
                    let _e4025_ = phi_14924_;
                    let _e4026_ = select(_e4006_, -1000000000000000000f, _e4025_);
                    if (_e4011_ != _e4011_) {
                        phi_14939_ = true;
                    } else {
                        phi_14939_ = (1000000000000000000f <= _e4011_);
                    }
                    let _e4030_ = phi_14939_;
                    let _e4031_ = select(_e4011_, 1000000000000000000f, _e4030_);
                    if (_e4016_ != _e4016_) {
                        phi_14954_ = true;
                    } else {
                        phi_14954_ = (1000000000000000000f <= _e4016_);
                    }
                    let _e4035_ = phi_14954_;
                    let _e4036_ = select(_e4016_, 1000000000000000000f, _e4035_);
                    if (_e4021_ != _e4021_) {
                        phi_14969_ = true;
                    } else {
                        phi_14969_ = (1000000000000000000f <= _e4021_);
                    }
                    let _e4040_ = phi_14969_;
                    let _e4041_ = select(_e4021_, 1000000000000000000f, _e4040_);
                    if (_e4026_ != _e4026_) {
                        phi_14984_ = true;
                    } else {
                        phi_14984_ = (1000000000000000000f <= _e4026_);
                    }
                    let _e4045_ = phi_14984_;
                    let _e4046_ = select(_e4026_, 1000000000000000000f, _e4045_);
                    let _e4047_ = array<f32, 4>(_e4031_, _e4036_, _e4041_, _e4046_);
                    let _e4048_ = (_e3971_ != _e3971_);
                    if _e4048_ {
                        phi_14999_ = true;
                    } else {
                        phi_14999_ = (_e4031_ <= _e3971_);
                    }
                    let _e4051_ = phi_14999_;
                    let _e4052_ = select(_e3971_, _e4031_, _e4051_);
                    let _e4053_ = (_e3976_ != _e3976_);
                    if _e4053_ {
                        phi_15014_ = true;
                    } else {
                        phi_15014_ = (_e4036_ <= _e3976_);
                    }
                    let _e4056_ = phi_15014_;
                    let _e4057_ = select(_e3976_, _e4036_, _e4056_);
                    let _e4058_ = (_e3981_ != _e3981_);
                    if _e4058_ {
                        phi_15029_ = true;
                    } else {
                        phi_15029_ = (_e4041_ <= _e3981_);
                    }
                    let _e4061_ = phi_15029_;
                    let _e4062_ = select(_e3981_, _e4041_, _e4061_);
                    let _e4063_ = (_e3986_ != _e3986_);
                    if _e4063_ {
                        phi_15044_ = true;
                    } else {
                        phi_15044_ = (_e4046_ <= _e3986_);
                    }
                    let _e4066_ = phi_15044_;
                    let _e4067_ = select(_e3986_, _e4046_, _e4066_);
                    if _e4048_ {
                        phi_15059_ = true;
                    } else {
                        phi_15059_ = (_e4031_ >= _e3971_);
                    }
                    let _e4070_ = phi_15059_;
                    let _e4071_ = select(_e3971_, _e4031_, _e4070_);
                    if _e4053_ {
                        phi_15074_ = true;
                    } else {
                        phi_15074_ = (_e4036_ >= _e3976_);
                    }
                    let _e4074_ = phi_15074_;
                    let _e4075_ = select(_e3976_, _e4036_, _e4074_);
                    if _e4058_ {
                        phi_15089_ = true;
                    } else {
                        phi_15089_ = (_e4041_ >= _e3981_);
                    }
                    let _e4078_ = phi_15089_;
                    let _e4079_ = select(_e3981_, _e4041_, _e4078_);
                    if _e4063_ {
                        phi_15104_ = true;
                    } else {
                        phi_15104_ = (_e4046_ >= _e3986_);
                    }
                    let _e4082_ = phi_15104_;
                    let _e4083_ = select(_e3986_, _e4046_, _e4082_);
                    let _e4084_ = (_e4052_ != _e4052_);
                    if _e4084_ {
                        phi_15119_ = true;
                    } else {
                        phi_15119_ = (-1f >= _e4052_);
                    }
                    let _e4087_ = phi_15119_;
                    let _e4088_ = select(_e4052_, -1f, _e4087_);
                    let _e4089_ = (_e4057_ != _e4057_);
                    if _e4089_ {
                        phi_15134_ = true;
                    } else {
                        phi_15134_ = (-1f >= _e4057_);
                    }
                    let _e4092_ = phi_15134_;
                    let _e4093_ = select(_e4057_, -1f, _e4092_);
                    let _e4094_ = (_e4062_ != _e4062_);
                    if _e4094_ {
                        phi_15149_ = true;
                    } else {
                        phi_15149_ = (-1f >= _e4062_);
                    }
                    let _e4097_ = phi_15149_;
                    let _e4098_ = select(_e4062_, -1f, _e4097_);
                    let _e4099_ = (_e4067_ != _e4067_);
                    if _e4099_ {
                        phi_15164_ = true;
                    } else {
                        phi_15164_ = (-1f >= _e4067_);
                    }
                    let _e4102_ = phi_15164_;
                    let _e4103_ = select(_e4067_, -1f, _e4102_);
                    if (_e4088_ != _e4088_) {
                        phi_15179_ = true;
                    } else {
                        phi_15179_ = (1f <= _e4088_);
                    }
                    let _e4107_ = phi_15179_;
                    let _e4108_ = select(_e4088_, 1f, _e4107_);
                    if (_e4093_ != _e4093_) {
                        phi_15194_ = true;
                    } else {
                        phi_15194_ = (1f <= _e4093_);
                    }
                    let _e4112_ = phi_15194_;
                    let _e4113_ = select(_e4093_, 1f, _e4112_);
                    if (_e4098_ != _e4098_) {
                        phi_15209_ = true;
                    } else {
                        phi_15209_ = (1f <= _e4098_);
                    }
                    let _e4117_ = phi_15209_;
                    let _e4118_ = select(_e4098_, 1f, _e4117_);
                    if (_e4103_ != _e4103_) {
                        phi_15224_ = true;
                    } else {
                        phi_15224_ = (1f <= _e4103_);
                    }
                    let _e4122_ = phi_15224_;
                    let _e4123_ = select(_e4103_, 1f, _e4122_);
                    let _e4124_ = (_e4071_ != _e4071_);
                    if _e4124_ {
                        phi_15239_ = true;
                    } else {
                        phi_15239_ = (-1f >= _e4071_);
                    }
                    let _e4127_ = phi_15239_;
                    let _e4128_ = select(_e4071_, -1f, _e4127_);
                    let _e4129_ = (_e4075_ != _e4075_);
                    if _e4129_ {
                        phi_15254_ = true;
                    } else {
                        phi_15254_ = (-1f >= _e4075_);
                    }
                    let _e4132_ = phi_15254_;
                    let _e4133_ = select(_e4075_, -1f, _e4132_);
                    let _e4134_ = (_e4079_ != _e4079_);
                    if _e4134_ {
                        phi_15269_ = true;
                    } else {
                        phi_15269_ = (-1f >= _e4079_);
                    }
                    let _e4137_ = phi_15269_;
                    let _e4138_ = select(_e4079_, -1f, _e4137_);
                    let _e4139_ = (_e4083_ != _e4083_);
                    if _e4139_ {
                        phi_15284_ = true;
                    } else {
                        phi_15284_ = (-1f >= _e4083_);
                    }
                    let _e4142_ = phi_15284_;
                    let _e4143_ = select(_e4083_, -1f, _e4142_);
                    if (_e4128_ != _e4128_) {
                        phi_15299_ = true;
                    } else {
                        phi_15299_ = (1f <= _e4128_);
                    }
                    let _e4147_ = phi_15299_;
                    let _e4148_ = select(_e4128_, 1f, _e4147_);
                    if (_e4133_ != _e4133_) {
                        phi_15314_ = true;
                    } else {
                        phi_15314_ = (1f <= _e4133_);
                    }
                    let _e4152_ = phi_15314_;
                    let _e4153_ = select(_e4133_, 1f, _e4152_);
                    if (_e4138_ != _e4138_) {
                        phi_15329_ = true;
                    } else {
                        phi_15329_ = (1f <= _e4138_);
                    }
                    let _e4157_ = phi_15329_;
                    let _e4158_ = select(_e4138_, 1f, _e4157_);
                    if (_e4143_ != _e4143_) {
                        phi_15344_ = true;
                    } else {
                        phi_15344_ = (1f <= _e4143_);
                    }
                    let _e4162_ = phi_15344_;
                    let _e4163_ = select(_e4143_, 1f, _e4162_);
                    let _e4168_ = (0.5f * (_e4148_ + _e4108_));
                    let _e4169_ = (0.5f * (_e4153_ + _e4113_));
                    let _e4170_ = (0.5f * (_e4158_ + _e4118_));
                    let _e4171_ = (0.5f * (_e4163_ + _e4123_));
                    let _e4172_ = (_e4148_ - _e4108_);
                    let _e4173_ = (_e4153_ - _e4113_);
                    let _e4174_ = (_e4158_ - _e4118_);
                    let _e4175_ = (_e4163_ - _e4123_);
                    let _e4176_ = (0.5f * _e4172_);
                    let _e4177_ = (0.5f * _e4173_);
                    let _e4178_ = (0.5f * _e4174_);
                    let _e4179_ = (0.5f * _e4175_);
                    let _e4180_ = (_e4168_ * _e4168_);
                    let _e4181_ = (_e4169_ * _e4169_);
                    let _e4182_ = (_e4170_ * _e4170_);
                    let _e4183_ = (_e4171_ * _e4171_);
                    let _e4184_ = (_e4176_ * _e4176_);
                    let _e4185_ = (_e4177_ * _e4177_);
                    let _e4186_ = (_e4178_ * _e4178_);
                    let _e4187_ = (_e4179_ * _e4179_);
                    if (_e4172_ != _e4172_) {
                        phi_15359_ = true;
                    } else {
                        phi_15359_ = (0f >= _e4172_);
                    }
                    let _e4191_ = phi_15359_;
                    let _e4192_ = select(_e4172_, 0f, _e4191_);
                    if (_e4173_ != _e4173_) {
                        phi_15374_ = true;
                    } else {
                        phi_15374_ = (0f >= _e4173_);
                    }
                    let _e4196_ = phi_15374_;
                    let _e4197_ = select(_e4173_, 0f, _e4196_);
                    if (_e4174_ != _e4174_) {
                        phi_15389_ = true;
                    } else {
                        phi_15389_ = (0f >= _e4174_);
                    }
                    let _e4201_ = phi_15389_;
                    let _e4202_ = select(_e4174_, 0f, _e4201_);
                    if (_e4175_ != _e4175_) {
                        phi_15404_ = true;
                    } else {
                        phi_15404_ = (0f >= _e4175_);
                    }
                    let _e4206_ = phi_15404_;
                    let _e4207_ = select(_e4175_, 0f, _e4206_);
                    let _e4208_ = (_e4071_ - _e4052_);
                    let _e4209_ = (_e4075_ - _e4057_);
                    let _e4210_ = (_e4079_ - _e4062_);
                    let _e4211_ = (_e4083_ - _e4067_);
                    if _e4124_ {
                        phi_15419_ = true;
                    } else {
                        phi_15419_ = (-1f <= _e4071_);
                    }
                    let _e4214_ = phi_15419_;
                    if _e4129_ {
                        phi_15434_ = true;
                    } else {
                        phi_15434_ = (-1f <= _e4075_);
                    }
                    let _e4218_ = phi_15434_;
                    if _e4134_ {
                        phi_15449_ = true;
                    } else {
                        phi_15449_ = (-1f <= _e4079_);
                    }
                    let _e4222_ = phi_15449_;
                    if _e4139_ {
                        phi_15464_ = true;
                    } else {
                        phi_15464_ = (-1f <= _e4083_);
                    }
                    let _e4226_ = phi_15464_;
                    let _e4228_ = (select(_e4071_, -1f, _e4214_) - _e4052_);
                    let _e4229_ = (select(_e4075_, -1f, _e4218_) - _e4057_);
                    let _e4230_ = (select(_e4079_, -1f, _e4222_) - _e4062_);
                    let _e4231_ = (select(_e4083_, -1f, _e4226_) - _e4067_);
                    if (_e4228_ != _e4228_) {
                        phi_15479_ = true;
                    } else {
                        phi_15479_ = (0f >= _e4228_);
                    }
                    let _e4235_ = phi_15479_;
                    if (_e4229_ != _e4229_) {
                        phi_15494_ = true;
                    } else {
                        phi_15494_ = (0f >= _e4229_);
                    }
                    let _e4240_ = phi_15494_;
                    if (_e4230_ != _e4230_) {
                        phi_15509_ = true;
                    } else {
                        phi_15509_ = (0f >= _e4230_);
                    }
                    let _e4245_ = phi_15509_;
                    if (_e4231_ != _e4231_) {
                        phi_15524_ = true;
                    } else {
                        phi_15524_ = (0f >= _e4231_);
                    }
                    let _e4250_ = phi_15524_;
                    if _e4084_ {
                        phi_15539_ = true;
                    } else {
                        phi_15539_ = (1f >= _e4052_);
                    }
                    let _e4254_ = phi_15539_;
                    if _e4089_ {
                        phi_15554_ = true;
                    } else {
                        phi_15554_ = (1f >= _e4057_);
                    }
                    let _e4258_ = phi_15554_;
                    if _e4094_ {
                        phi_15569_ = true;
                    } else {
                        phi_15569_ = (1f >= _e4062_);
                    }
                    let _e4262_ = phi_15569_;
                    if _e4099_ {
                        phi_15584_ = true;
                    } else {
                        phi_15584_ = (1f >= _e4067_);
                    }
                    let _e4266_ = phi_15584_;
                    let _e4268_ = (_e4071_ - select(_e4052_, 1f, _e4254_));
                    let _e4269_ = (_e4075_ - select(_e4057_, 1f, _e4258_));
                    let _e4270_ = (_e4079_ - select(_e4062_, 1f, _e4262_));
                    let _e4271_ = (_e4083_ - select(_e4067_, 1f, _e4266_));
                    if (_e4268_ != _e4268_) {
                        phi_15599_ = true;
                    } else {
                        phi_15599_ = (0f >= _e4268_);
                    }
                    let _e4275_ = phi_15599_;
                    if (_e4269_ != _e4269_) {
                        phi_15614_ = true;
                    } else {
                        phi_15614_ = (0f >= _e4269_);
                    }
                    let _e4280_ = phi_15614_;
                    if (_e4270_ != _e4270_) {
                        phi_15629_ = true;
                    } else {
                        phi_15629_ = (0f >= _e4270_);
                    }
                    let _e4285_ = phi_15629_;
                    if (_e4271_ != _e4271_) {
                        phi_15644_ = true;
                    } else {
                        phi_15644_ = (0f >= _e4271_);
                    }
                    let _e4290_ = phi_15644_;
                    let _e4294_ = bitcast<f32>((bitcast<u32>(_e3971_) & 2147483647u));
                    let _e4297_ = bitcast<f32>((bitcast<u32>(_e3976_) & 2147483647u));
                    let _e4300_ = bitcast<f32>((bitcast<u32>(_e3981_) & 2147483647u));
                    let _e4303_ = bitcast<f32>((bitcast<u32>(_e3986_) & 2147483647u));
                    let _e4306_ = bitcast<f32>((bitcast<u32>(_e4031_) & 2147483647u));
                    let _e4309_ = bitcast<f32>((bitcast<u32>(_e4036_) & 2147483647u));
                    let _e4312_ = bitcast<f32>((bitcast<u32>(_e4041_) & 2147483647u));
                    let _e4315_ = bitcast<f32>((bitcast<u32>(_e4046_) & 2147483647u));
                    if (_e4294_ != _e4294_) {
                        phi_15659_ = true;
                    } else {
                        phi_15659_ = (_e4306_ >= _e4294_);
                    }
                    let _e4319_ = phi_15659_;
                    let _e4320_ = select(_e4294_, _e4306_, _e4319_);
                    if (_e4297_ != _e4297_) {
                        phi_15674_ = true;
                    } else {
                        phi_15674_ = (_e4309_ >= _e4297_);
                    }
                    let _e4324_ = phi_15674_;
                    let _e4325_ = select(_e4297_, _e4309_, _e4324_);
                    if (_e4300_ != _e4300_) {
                        phi_15689_ = true;
                    } else {
                        phi_15689_ = (_e4312_ >= _e4300_);
                    }
                    let _e4329_ = phi_15689_;
                    let _e4330_ = select(_e4300_, _e4312_, _e4329_);
                    if (_e4303_ != _e4303_) {
                        phi_15704_ = true;
                    } else {
                        phi_15704_ = (_e4315_ >= _e4303_);
                    }
                    let _e4334_ = phi_15704_;
                    let _e4335_ = select(_e4303_, _e4315_, _e4334_);
                    let _e4352_ = (_e4320_ <= 0f);
                    let _e4353_ = select((_e4168_ * (1.5f - (0.5f * (_e4180_ + _e4184_)))), 0f, _e4352_);
                    let _e4354_ = (_e4325_ <= 0f);
                    let _e4355_ = select((_e4169_ * (1.5f - (0.5f * (_e4181_ + _e4185_)))), 0f, _e4354_);
                    let _e4356_ = (_e4330_ <= 0f);
                    let _e4357_ = select((_e4170_ * (1.5f - (0.5f * (_e4182_ + _e4186_)))), 0f, _e4356_);
                    let _e4358_ = (_e4335_ <= 0f);
                    let _e4359_ = select((_e4171_ * (1.5f - (0.5f * (_e4183_ + _e4187_)))), 0f, _e4358_);
                    let _e4401_ = select(((_e4180_ + (0.33333334f * _e4184_)) - (((_e4180_ * _e4180_) + ((2f * _e4180_) * _e4184_)) + (0.2f * (_e4184_ * _e4184_)))), 0f, _e4352_);
                    let _e4402_ = select(((_e4181_ + (0.33333334f * _e4185_)) - (((_e4181_ * _e4181_) + ((2f * _e4181_) * _e4185_)) + (0.2f * (_e4185_ * _e4185_)))), 0f, _e4354_);
                    let _e4403_ = select(((_e4182_ + (0.33333334f * _e4186_)) - (((_e4182_ * _e4182_) + ((2f * _e4182_) * _e4186_)) + (0.2f * (_e4186_ * _e4186_)))), 0f, _e4356_);
                    let _e4404_ = select(((_e4183_ + (0.33333334f * _e4187_)) - (((_e4183_ * _e4183_) + ((2f * _e4183_) * _e4187_)) + (0.2f * (_e4187_ * _e4187_)))), 0f, _e4358_);
                    if (_e4320_ <= 1f) {
                        if (_e4325_ <= 1f) {
                            let _e4408_ = (_e4330_ <= 1f);
                            if _e4408_ {
                                phi_8818_ = (_e4335_ <= 1f);
                            } else {
                                phi_8818_ = bool();
                            }
                            let _e4411_ = phi_8818_;
                            phi_8820_ = _e4411_;
                            phi_8821_ = select(true, false, _e4408_);
                        } else {
                            phi_8820_ = bool();
                            phi_8821_ = true;
                        }
                        let _e4414_ = phi_8820_;
                        let _e4416_ = phi_8821_;
                        phi_8822_ = _e4414_;
                        phi_8823_ = _e4416_;
                    } else {
                        phi_8822_ = bool();
                        phi_8823_ = true;
                    }
                    let _e4418_ = phi_8822_;
                    let _e4420_ = phi_8823_;
                    if select(_e4418_, false, _e4420_) {
                        phi_9201_ = type_33_(type_24_(array<f32, 4>(_e4353_, _e4355_, _e4357_, _e4359_)), type_24_(array<f32, 4>(_e4401_, _e4402_, _e4403_, _e4404_)), type_24_(_e4047_));
                    } else {
                        if (_e4208_ != _e4208_) {
                            phi_15719_ = true;
                        } else {
                            phi_15719_ = (0.000001f >= _e4208_);
                        }
                        let _e4425_ = phi_15719_;
                        let _e4426_ = select(_e4208_, 0.000001f, _e4425_);
                        if (_e4209_ != _e4209_) {
                            phi_15734_ = true;
                        } else {
                            phi_15734_ = (0.000001f >= _e4209_);
                        }
                        let _e4430_ = phi_15734_;
                        let _e4431_ = select(_e4209_, 0.000001f, _e4430_);
                        if (_e4210_ != _e4210_) {
                            phi_15749_ = true;
                        } else {
                            phi_15749_ = (0.000001f >= _e4210_);
                        }
                        let _e4435_ = phi_15749_;
                        let _e4436_ = select(_e4210_, 0.000001f, _e4435_);
                        if (_e4211_ != _e4211_) {
                            phi_15764_ = true;
                        } else {
                            phi_15764_ = (0.000001f >= _e4211_);
                        }
                        let _e4440_ = phi_15764_;
                        let _e4441_ = select(_e4211_, 0.000001f, _e4440_);
                        let _e4444_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4426_)));
                        let _e4447_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4431_)));
                        let _e4450_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4436_)));
                        let _e4453_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4441_)));
                        let _e4463_ = (_e4444_ + (_e4444_ * (1f - (_e4426_ * _e4444_))));
                        let _e4465_ = (_e4447_ + (_e4447_ * (1f - (_e4431_ * _e4447_))));
                        let _e4467_ = (_e4450_ + (_e4450_ * (1f - (_e4436_ * _e4450_))));
                        let _e4469_ = (_e4453_ + (_e4453_ * (1f - (_e4441_ * _e4453_))));
                        let _e4479_ = (_e4463_ + (_e4463_ * (1f - (_e4426_ * _e4463_))));
                        let _e4481_ = (_e4465_ + (_e4465_ * (1f - (_e4431_ * _e4465_))));
                        let _e4483_ = (_e4467_ + (_e4467_ * (1f - (_e4436_ * _e4467_))));
                        let _e4485_ = (_e4469_ + (_e4469_ * (1f - (_e4441_ * _e4469_))));
                        let _e4538_ = (select((select(0f, 1f, (-1f <= _e4052_)) * select(0f, 1f, (_e4071_ <= 1f))), 1f, (_e4208_ <= 0.000001f)) <= 0f);
                        let _e4540_ = (select((select(0f, 1f, (-1f <= _e4057_)) * select(0f, 1f, (_e4075_ <= 1f))), 1f, (_e4209_ <= 0.000001f)) <= 0f);
                        let _e4542_ = (select((select(0f, 1f, (-1f <= _e4062_)) * select(0f, 1f, (_e4079_ <= 1f))), 1f, (_e4210_ <= 0.000001f)) <= 0f);
                        let _e4544_ = (select((select(0f, 1f, (-1f <= _e4067_)) * select(0f, 1f, (_e4083_ <= 1f))), 1f, (_e4211_ <= 0.000001f)) <= 0f);
                        let _e4550_ = (1f <= _e4052_);
                        let _e4552_ = (1f <= _e4057_);
                        let _e4554_ = (1f <= _e4062_);
                        let _e4556_ = (1f <= _e4067_);
                        let _e4558_ = (_e4071_ <= -1f);
                        let _e4560_ = (_e4075_ <= -1f);
                        let _e4562_ = (_e4079_ <= -1f);
                        let _e4564_ = (_e4083_ <= -1f);
                        phi_9201_ = type_33_(type_24_(array<f32, 4>(select(select(select(_e4353_, (((select(_e4268_, 0f, _e4275_) - select(_e4228_, 0f, _e4235_)) + (_e4192_ * _e4353_)) * _e4479_), _e4538_), 1f, _e4550_), -1f, _e4558_), select(select(select(_e4355_, (((select(_e4269_, 0f, _e4280_) - select(_e4229_, 0f, _e4240_)) + (_e4197_ * _e4355_)) * _e4481_), _e4540_), 1f, _e4552_), -1f, _e4560_), select(select(select(_e4357_, (((select(_e4270_, 0f, _e4285_) - select(_e4230_, 0f, _e4245_)) + (_e4202_ * _e4357_)) * _e4483_), _e4542_), 1f, _e4554_), -1f, _e4562_), select(select(select(_e4359_, (((select(_e4271_, 0f, _e4290_) - select(_e4231_, 0f, _e4250_)) + (_e4207_ * _e4359_)) * _e4485_), _e4544_), 1f, _e4556_), -1f, _e4564_))), type_24_(array<f32, 4>(select(select(select(_e4401_, ((_e4192_ * _e4401_) * _e4479_), _e4538_), 0f, _e4550_), 0f, _e4558_), select(select(select(_e4402_, ((_e4197_ * _e4402_) * _e4481_), _e4540_), 0f, _e4552_), 0f, _e4560_), select(select(select(_e4403_, ((_e4202_ * _e4403_) * _e4483_), _e4542_), 0f, _e4554_), 0f, _e4562_), select(select(select(_e4404_, ((_e4207_ * _e4404_) * _e4485_), _e4544_), 0f, _e4556_), 0f, _e4564_))), type_24_(_e4047_));
                    }
                    let _e4585_ = phi_9201_;
                    if (_e417_ <= 0f) {
                        if (_e437_ <= 0f) {
                            let _e4589_ = (_e457_ <= 0f);
                            if _e4589_ {
                                phi_9218_ = (_e477_ <= 0f);
                            } else {
                                phi_9218_ = bool();
                            }
                            let _e4592_ = phi_9218_;
                            phi_9220_ = _e4592_;
                            phi_9221_ = select(true, false, _e4589_);
                        } else {
                            phi_9220_ = bool();
                            phi_9221_ = true;
                        }
                        let _e4595_ = phi_9220_;
                        let _e4597_ = phi_9221_;
                        phi_9222_ = _e4595_;
                        phi_9223_ = _e4597_;
                    } else {
                        phi_9222_ = bool();
                        phi_9223_ = true;
                    }
                    let _e4599_ = phi_9222_;
                    let _e4601_ = phi_9223_;
                    if select(_e4599_, false, _e4601_) {
                        phi_11111_ = _e4585_.member;
                        phi_11112_ = _e3917_.member;
                    } else {
                        if _e3279_ {
                            phi_15779_ = true;
                        } else {
                            phi_15779_ = (-1000000000000000000f >= _e3272_);
                        }
                        let _e4605_ = phi_15779_;
                        let _e4606_ = select(_e3272_, -1000000000000000000f, _e4605_);
                        if _e3284_ {
                            phi_15794_ = true;
                        } else {
                            phi_15794_ = (-1000000000000000000f >= _e3274_);
                        }
                        let _e4609_ = phi_15794_;
                        let _e4610_ = select(_e3274_, -1000000000000000000f, _e4609_);
                        if _e3289_ {
                            phi_15809_ = true;
                        } else {
                            phi_15809_ = (-1000000000000000000f >= _e3276_);
                        }
                        let _e4613_ = phi_15809_;
                        let _e4614_ = select(_e3276_, -1000000000000000000f, _e4613_);
                        if _e3294_ {
                            phi_15824_ = true;
                        } else {
                            phi_15824_ = (-1000000000000000000f >= _e3278_);
                        }
                        let _e4617_ = phi_15824_;
                        let _e4618_ = select(_e3278_, -1000000000000000000f, _e4617_);
                        if (_e4606_ != _e4606_) {
                            phi_15839_ = true;
                        } else {
                            phi_15839_ = (1000000000000000000f <= _e4606_);
                        }
                        let _e4622_ = phi_15839_;
                        let _e4623_ = select(_e4606_, 1000000000000000000f, _e4622_);
                        if (_e4610_ != _e4610_) {
                            phi_15854_ = true;
                        } else {
                            phi_15854_ = (1000000000000000000f <= _e4610_);
                        }
                        let _e4627_ = phi_15854_;
                        let _e4628_ = select(_e4610_, 1000000000000000000f, _e4627_);
                        if (_e4614_ != _e4614_) {
                            phi_15869_ = true;
                        } else {
                            phi_15869_ = (1000000000000000000f <= _e4614_);
                        }
                        let _e4632_ = phi_15869_;
                        let _e4633_ = select(_e4614_, 1000000000000000000f, _e4632_);
                        if (_e4618_ != _e4618_) {
                            phi_15884_ = true;
                        } else {
                            phi_15884_ = (1000000000000000000f <= _e4618_);
                        }
                        let _e4637_ = phi_15884_;
                        let _e4638_ = select(_e4618_, 1000000000000000000f, _e4637_);
                        if _e3339_ {
                            phi_15899_ = true;
                        } else {
                            phi_15899_ = (-1000000000000000000f >= _e3332_);
                        }
                        let _e4641_ = phi_15899_;
                        let _e4642_ = select(_e3332_, -1000000000000000000f, _e4641_);
                        if _e3344_ {
                            phi_15914_ = true;
                        } else {
                            phi_15914_ = (-1000000000000000000f >= _e3334_);
                        }
                        let _e4645_ = phi_15914_;
                        let _e4646_ = select(_e3334_, -1000000000000000000f, _e4645_);
                        if _e3349_ {
                            phi_15929_ = true;
                        } else {
                            phi_15929_ = (-1000000000000000000f >= _e3336_);
                        }
                        let _e4649_ = phi_15929_;
                        let _e4650_ = select(_e3336_, -1000000000000000000f, _e4649_);
                        if _e3354_ {
                            phi_15944_ = true;
                        } else {
                            phi_15944_ = (-1000000000000000000f >= _e3338_);
                        }
                        let _e4653_ = phi_15944_;
                        let _e4654_ = select(_e3338_, -1000000000000000000f, _e4653_);
                        if (_e4642_ != _e4642_) {
                            phi_15959_ = true;
                        } else {
                            phi_15959_ = (1000000000000000000f <= _e4642_);
                        }
                        let _e4658_ = phi_15959_;
                        let _e4659_ = select(_e4642_, 1000000000000000000f, _e4658_);
                        if (_e4646_ != _e4646_) {
                            phi_15974_ = true;
                        } else {
                            phi_15974_ = (1000000000000000000f <= _e4646_);
                        }
                        let _e4663_ = phi_15974_;
                        let _e4664_ = select(_e4646_, 1000000000000000000f, _e4663_);
                        if (_e4650_ != _e4650_) {
                            phi_15989_ = true;
                        } else {
                            phi_15989_ = (1000000000000000000f <= _e4650_);
                        }
                        let _e4668_ = phi_15989_;
                        let _e4669_ = select(_e4650_, 1000000000000000000f, _e4668_);
                        if (_e4654_ != _e4654_) {
                            phi_16004_ = true;
                        } else {
                            phi_16004_ = (1000000000000000000f <= _e4654_);
                        }
                        let _e4673_ = phi_16004_;
                        let _e4674_ = select(_e4654_, 1000000000000000000f, _e4673_);
                        let _e4675_ = (_e4623_ != _e4623_);
                        if _e4675_ {
                            phi_16019_ = true;
                        } else {
                            phi_16019_ = (_e4659_ <= _e4623_);
                        }
                        let _e4678_ = phi_16019_;
                        let _e4679_ = select(_e4623_, _e4659_, _e4678_);
                        let _e4680_ = (_e4628_ != _e4628_);
                        if _e4680_ {
                            phi_16034_ = true;
                        } else {
                            phi_16034_ = (_e4664_ <= _e4628_);
                        }
                        let _e4683_ = phi_16034_;
                        let _e4684_ = select(_e4628_, _e4664_, _e4683_);
                        let _e4685_ = (_e4633_ != _e4633_);
                        if _e4685_ {
                            phi_16049_ = true;
                        } else {
                            phi_16049_ = (_e4669_ <= _e4633_);
                        }
                        let _e4688_ = phi_16049_;
                        let _e4689_ = select(_e4633_, _e4669_, _e4688_);
                        let _e4690_ = (_e4638_ != _e4638_);
                        if _e4690_ {
                            phi_16064_ = true;
                        } else {
                            phi_16064_ = (_e4674_ <= _e4638_);
                        }
                        let _e4693_ = phi_16064_;
                        let _e4694_ = select(_e4638_, _e4674_, _e4693_);
                        if _e4675_ {
                            phi_16079_ = true;
                        } else {
                            phi_16079_ = (_e4659_ >= _e4623_);
                        }
                        let _e4697_ = phi_16079_;
                        let _e4698_ = select(_e4623_, _e4659_, _e4697_);
                        if _e4680_ {
                            phi_16094_ = true;
                        } else {
                            phi_16094_ = (_e4664_ >= _e4628_);
                        }
                        let _e4701_ = phi_16094_;
                        let _e4702_ = select(_e4628_, _e4664_, _e4701_);
                        if _e4685_ {
                            phi_16109_ = true;
                        } else {
                            phi_16109_ = (_e4669_ >= _e4633_);
                        }
                        let _e4705_ = phi_16109_;
                        let _e4706_ = select(_e4633_, _e4669_, _e4705_);
                        if _e4690_ {
                            phi_16124_ = true;
                        } else {
                            phi_16124_ = (_e4674_ >= _e4638_);
                        }
                        let _e4709_ = phi_16124_;
                        let _e4710_ = select(_e4638_, _e4674_, _e4709_);
                        if (_e4679_ != _e4679_) {
                            phi_16139_ = true;
                        } else {
                            phi_16139_ = (-1f >= _e4679_);
                        }
                        let _e4714_ = phi_16139_;
                        let _e4715_ = select(_e4679_, -1f, _e4714_);
                        if (_e4684_ != _e4684_) {
                            phi_16154_ = true;
                        } else {
                            phi_16154_ = (-1f >= _e4684_);
                        }
                        let _e4719_ = phi_16154_;
                        let _e4720_ = select(_e4684_, -1f, _e4719_);
                        if (_e4689_ != _e4689_) {
                            phi_16169_ = true;
                        } else {
                            phi_16169_ = (-1f >= _e4689_);
                        }
                        let _e4724_ = phi_16169_;
                        let _e4725_ = select(_e4689_, -1f, _e4724_);
                        if (_e4694_ != _e4694_) {
                            phi_16184_ = true;
                        } else {
                            phi_16184_ = (-1f >= _e4694_);
                        }
                        let _e4729_ = phi_16184_;
                        let _e4730_ = select(_e4694_, -1f, _e4729_);
                        if (_e4715_ != _e4715_) {
                            phi_16199_ = true;
                        } else {
                            phi_16199_ = (1f <= _e4715_);
                        }
                        let _e4734_ = phi_16199_;
                        let _e4735_ = select(_e4715_, 1f, _e4734_);
                        if (_e4720_ != _e4720_) {
                            phi_16214_ = true;
                        } else {
                            phi_16214_ = (1f <= _e4720_);
                        }
                        let _e4739_ = phi_16214_;
                        let _e4740_ = select(_e4720_, 1f, _e4739_);
                        if (_e4725_ != _e4725_) {
                            phi_16229_ = true;
                        } else {
                            phi_16229_ = (1f <= _e4725_);
                        }
                        let _e4744_ = phi_16229_;
                        let _e4745_ = select(_e4725_, 1f, _e4744_);
                        if (_e4730_ != _e4730_) {
                            phi_16244_ = true;
                        } else {
                            phi_16244_ = (1f <= _e4730_);
                        }
                        let _e4749_ = phi_16244_;
                        let _e4750_ = select(_e4730_, 1f, _e4749_);
                        if (_e4698_ != _e4698_) {
                            phi_16259_ = true;
                        } else {
                            phi_16259_ = (-1f >= _e4698_);
                        }
                        let _e4754_ = phi_16259_;
                        let _e4755_ = select(_e4698_, -1f, _e4754_);
                        if (_e4702_ != _e4702_) {
                            phi_16274_ = true;
                        } else {
                            phi_16274_ = (-1f >= _e4702_);
                        }
                        let _e4759_ = phi_16274_;
                        let _e4760_ = select(_e4702_, -1f, _e4759_);
                        if (_e4706_ != _e4706_) {
                            phi_16289_ = true;
                        } else {
                            phi_16289_ = (-1f >= _e4706_);
                        }
                        let _e4764_ = phi_16289_;
                        let _e4765_ = select(_e4706_, -1f, _e4764_);
                        if (_e4710_ != _e4710_) {
                            phi_16304_ = true;
                        } else {
                            phi_16304_ = (-1f >= _e4710_);
                        }
                        let _e4769_ = phi_16304_;
                        let _e4770_ = select(_e4710_, -1f, _e4769_);
                        if (_e4755_ != _e4755_) {
                            phi_16319_ = true;
                        } else {
                            phi_16319_ = (1f <= _e4755_);
                        }
                        let _e4774_ = phi_16319_;
                        let _e4775_ = select(_e4755_, 1f, _e4774_);
                        if (_e4760_ != _e4760_) {
                            phi_16334_ = true;
                        } else {
                            phi_16334_ = (1f <= _e4760_);
                        }
                        let _e4779_ = phi_16334_;
                        let _e4780_ = select(_e4760_, 1f, _e4779_);
                        if (_e4765_ != _e4765_) {
                            phi_16349_ = true;
                        } else {
                            phi_16349_ = (1f <= _e4765_);
                        }
                        let _e4784_ = phi_16349_;
                        let _e4785_ = select(_e4765_, 1f, _e4784_);
                        if (_e4770_ != _e4770_) {
                            phi_16364_ = true;
                        } else {
                            phi_16364_ = (1f <= _e4770_);
                        }
                        let _e4789_ = phi_16364_;
                        let _e4790_ = select(_e4770_, 1f, _e4789_);
                        let _e4795_ = (0.5f * (_e4775_ + _e4735_));
                        let _e4796_ = (0.5f * (_e4780_ + _e4740_));
                        let _e4797_ = (0.5f * (_e4785_ + _e4745_));
                        let _e4798_ = (0.5f * (_e4790_ + _e4750_));
                        let _e4799_ = (_e4775_ - _e4735_);
                        let _e4800_ = (_e4780_ - _e4740_);
                        let _e4801_ = (_e4785_ - _e4745_);
                        let _e4802_ = (_e4790_ - _e4750_);
                        let _e4803_ = (0.5f * _e4799_);
                        let _e4804_ = (0.5f * _e4800_);
                        let _e4805_ = (0.5f * _e4801_);
                        let _e4806_ = (0.5f * _e4802_);
                        let _e4807_ = (_e4795_ * _e4795_);
                        let _e4808_ = (_e4796_ * _e4796_);
                        let _e4809_ = (_e4797_ * _e4797_);
                        let _e4810_ = (_e4798_ * _e4798_);
                        let _e4811_ = (_e4803_ * _e4803_);
                        let _e4812_ = (_e4804_ * _e4804_);
                        let _e4813_ = (_e4805_ * _e4805_);
                        let _e4814_ = (_e4806_ * _e4806_);
                        if (_e4799_ != _e4799_) {
                            phi_16379_ = true;
                        } else {
                            phi_16379_ = (0f >= _e4799_);
                        }
                        let _e4818_ = phi_16379_;
                        if (_e4800_ != _e4800_) {
                            phi_16394_ = true;
                        } else {
                            phi_16394_ = (0f >= _e4800_);
                        }
                        let _e4823_ = phi_16394_;
                        if (_e4801_ != _e4801_) {
                            phi_16409_ = true;
                        } else {
                            phi_16409_ = (0f >= _e4801_);
                        }
                        let _e4828_ = phi_16409_;
                        if (_e4802_ != _e4802_) {
                            phi_16424_ = true;
                        } else {
                            phi_16424_ = (0f >= _e4802_);
                        }
                        let _e4833_ = phi_16424_;
                        let _e4835_ = (_e4698_ - _e4679_);
                        let _e4836_ = (_e4702_ - _e4684_);
                        let _e4837_ = (_e4706_ - _e4689_);
                        let _e4838_ = (_e4710_ - _e4694_);
                        let _e4841_ = bitcast<f32>((bitcast<u32>(_e4623_) & 2147483647u));
                        let _e4844_ = bitcast<f32>((bitcast<u32>(_e4628_) & 2147483647u));
                        let _e4847_ = bitcast<f32>((bitcast<u32>(_e4633_) & 2147483647u));
                        let _e4850_ = bitcast<f32>((bitcast<u32>(_e4638_) & 2147483647u));
                        let _e4853_ = bitcast<f32>((bitcast<u32>(_e4659_) & 2147483647u));
                        let _e4856_ = bitcast<f32>((bitcast<u32>(_e4664_) & 2147483647u));
                        let _e4859_ = bitcast<f32>((bitcast<u32>(_e4669_) & 2147483647u));
                        let _e4862_ = bitcast<f32>((bitcast<u32>(_e4674_) & 2147483647u));
                        if (_e4841_ != _e4841_) {
                            phi_16679_ = true;
                        } else {
                            phi_16679_ = (_e4853_ >= _e4841_);
                        }
                        let _e4866_ = phi_16679_;
                        if (_e4844_ != _e4844_) {
                            phi_16694_ = true;
                        } else {
                            phi_16694_ = (_e4856_ >= _e4844_);
                        }
                        let _e4871_ = phi_16694_;
                        if (_e4847_ != _e4847_) {
                            phi_16709_ = true;
                        } else {
                            phi_16709_ = (_e4859_ >= _e4847_);
                        }
                        let _e4876_ = phi_16709_;
                        if (_e4850_ != _e4850_) {
                            phi_16724_ = true;
                        } else {
                            phi_16724_ = (_e4862_ >= _e4850_);
                        }
                        let _e4881_ = phi_16724_;
                        let _e4887_ = (1f - (_e4807_ + _e4811_));
                        let _e4888_ = (1f - (_e4808_ + _e4812_));
                        let _e4889_ = (1f - (_e4809_ + _e4813_));
                        let _e4890_ = (1f - (_e4810_ + _e4814_));
                        let _e4916_ = select(((_e4795_ * _e4887_) * ((_e4887_ * _e4887_) + ((4f * _e4807_) * _e4811_))), 0f, (select(_e4841_, _e4853_, _e4866_) <= 0f));
                        let _e4918_ = select(((_e4796_ * _e4888_) * ((_e4888_ * _e4888_) + ((4f * _e4808_) * _e4812_))), 0f, (select(_e4844_, _e4856_, _e4871_) <= 0f));
                        let _e4920_ = select(((_e4797_ * _e4889_) * ((_e4889_ * _e4889_) + ((4f * _e4809_) * _e4813_))), 0f, (select(_e4847_, _e4859_, _e4876_) <= 0f));
                        let _e4922_ = select(((_e4798_ * _e4890_) * ((_e4890_ * _e4890_) + ((4f * _e4810_) * _e4814_))), 0f, (select(_e4850_, _e4862_, _e4881_) <= 0f));
                        if (_e4835_ != _e4835_) {
                            phi_16739_ = true;
                        } else {
                            phi_16739_ = (0.000001f >= _e4835_);
                        }
                        let _e4926_ = phi_16739_;
                        let _e4927_ = select(_e4835_, 0.000001f, _e4926_);
                        if (_e4836_ != _e4836_) {
                            phi_16754_ = true;
                        } else {
                            phi_16754_ = (0.000001f >= _e4836_);
                        }
                        let _e4931_ = phi_16754_;
                        let _e4932_ = select(_e4836_, 0.000001f, _e4931_);
                        if (_e4837_ != _e4837_) {
                            phi_16769_ = true;
                        } else {
                            phi_16769_ = (0.000001f >= _e4837_);
                        }
                        let _e4936_ = phi_16769_;
                        let _e4937_ = select(_e4837_, 0.000001f, _e4936_);
                        if (_e4838_ != _e4838_) {
                            phi_16784_ = true;
                        } else {
                            phi_16784_ = (0.000001f >= _e4838_);
                        }
                        let _e4941_ = phi_16784_;
                        let _e4942_ = select(_e4838_, 0.000001f, _e4941_);
                        let _e4945_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4927_)));
                        let _e4948_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4932_)));
                        let _e4951_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4937_)));
                        let _e4954_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4942_)));
                        let _e4964_ = (_e4945_ + (_e4945_ * (1f - (_e4927_ * _e4945_))));
                        let _e4966_ = (_e4948_ + (_e4948_ * (1f - (_e4932_ * _e4948_))));
                        let _e4968_ = (_e4951_ + (_e4951_ * (1f - (_e4937_ * _e4951_))));
                        let _e4970_ = (_e4954_ + (_e4954_ * (1f - (_e4942_ * _e4954_))));
                        if _e3947_ {
                            phi_16799_ = true;
                        } else {
                            phi_16799_ = (-1000000000000000000f >= _e3940_);
                        }
                        let _e5049_ = phi_16799_;
                        let _e5050_ = select(_e3940_, -1000000000000000000f, _e5049_);
                        if _e3952_ {
                            phi_16814_ = true;
                        } else {
                            phi_16814_ = (-1000000000000000000f >= _e3942_);
                        }
                        let _e5053_ = phi_16814_;
                        let _e5054_ = select(_e3942_, -1000000000000000000f, _e5053_);
                        if _e3957_ {
                            phi_16829_ = true;
                        } else {
                            phi_16829_ = (-1000000000000000000f >= _e3944_);
                        }
                        let _e5057_ = phi_16829_;
                        let _e5058_ = select(_e3944_, -1000000000000000000f, _e5057_);
                        if _e3962_ {
                            phi_16844_ = true;
                        } else {
                            phi_16844_ = (-1000000000000000000f >= _e3946_);
                        }
                        let _e5061_ = phi_16844_;
                        let _e5062_ = select(_e3946_, -1000000000000000000f, _e5061_);
                        if (_e5050_ != _e5050_) {
                            phi_16859_ = true;
                        } else {
                            phi_16859_ = (1000000000000000000f <= _e5050_);
                        }
                        let _e5066_ = phi_16859_;
                        let _e5067_ = select(_e5050_, 1000000000000000000f, _e5066_);
                        if (_e5054_ != _e5054_) {
                            phi_16874_ = true;
                        } else {
                            phi_16874_ = (1000000000000000000f <= _e5054_);
                        }
                        let _e5071_ = phi_16874_;
                        let _e5072_ = select(_e5054_, 1000000000000000000f, _e5071_);
                        if (_e5058_ != _e5058_) {
                            phi_16889_ = true;
                        } else {
                            phi_16889_ = (1000000000000000000f <= _e5058_);
                        }
                        let _e5076_ = phi_16889_;
                        let _e5077_ = select(_e5058_, 1000000000000000000f, _e5076_);
                        if (_e5062_ != _e5062_) {
                            phi_16904_ = true;
                        } else {
                            phi_16904_ = (1000000000000000000f <= _e5062_);
                        }
                        let _e5081_ = phi_16904_;
                        let _e5082_ = select(_e5062_, 1000000000000000000f, _e5081_);
                        if _e4007_ {
                            phi_16919_ = true;
                        } else {
                            phi_16919_ = (-1000000000000000000f >= _e4000_);
                        }
                        let _e5085_ = phi_16919_;
                        let _e5086_ = select(_e4000_, -1000000000000000000f, _e5085_);
                        if _e4012_ {
                            phi_16934_ = true;
                        } else {
                            phi_16934_ = (-1000000000000000000f >= _e4002_);
                        }
                        let _e5089_ = phi_16934_;
                        let _e5090_ = select(_e4002_, -1000000000000000000f, _e5089_);
                        if _e4017_ {
                            phi_16949_ = true;
                        } else {
                            phi_16949_ = (-1000000000000000000f >= _e4004_);
                        }
                        let _e5093_ = phi_16949_;
                        let _e5094_ = select(_e4004_, -1000000000000000000f, _e5093_);
                        if _e4022_ {
                            phi_16964_ = true;
                        } else {
                            phi_16964_ = (-1000000000000000000f >= _e4006_);
                        }
                        let _e5097_ = phi_16964_;
                        let _e5098_ = select(_e4006_, -1000000000000000000f, _e5097_);
                        if (_e5086_ != _e5086_) {
                            phi_16979_ = true;
                        } else {
                            phi_16979_ = (1000000000000000000f <= _e5086_);
                        }
                        let _e5102_ = phi_16979_;
                        let _e5103_ = select(_e5086_, 1000000000000000000f, _e5102_);
                        if (_e5090_ != _e5090_) {
                            phi_16994_ = true;
                        } else {
                            phi_16994_ = (1000000000000000000f <= _e5090_);
                        }
                        let _e5107_ = phi_16994_;
                        let _e5108_ = select(_e5090_, 1000000000000000000f, _e5107_);
                        if (_e5094_ != _e5094_) {
                            phi_17009_ = true;
                        } else {
                            phi_17009_ = (1000000000000000000f <= _e5094_);
                        }
                        let _e5112_ = phi_17009_;
                        let _e5113_ = select(_e5094_, 1000000000000000000f, _e5112_);
                        if (_e5098_ != _e5098_) {
                            phi_17024_ = true;
                        } else {
                            phi_17024_ = (1000000000000000000f <= _e5098_);
                        }
                        let _e5117_ = phi_17024_;
                        let _e5118_ = select(_e5098_, 1000000000000000000f, _e5117_);
                        let _e5119_ = (_e5067_ != _e5067_);
                        if _e5119_ {
                            phi_17039_ = true;
                        } else {
                            phi_17039_ = (_e5103_ <= _e5067_);
                        }
                        let _e5122_ = phi_17039_;
                        let _e5123_ = select(_e5067_, _e5103_, _e5122_);
                        let _e5124_ = (_e5072_ != _e5072_);
                        if _e5124_ {
                            phi_17054_ = true;
                        } else {
                            phi_17054_ = (_e5108_ <= _e5072_);
                        }
                        let _e5127_ = phi_17054_;
                        let _e5128_ = select(_e5072_, _e5108_, _e5127_);
                        let _e5129_ = (_e5077_ != _e5077_);
                        if _e5129_ {
                            phi_17069_ = true;
                        } else {
                            phi_17069_ = (_e5113_ <= _e5077_);
                        }
                        let _e5132_ = phi_17069_;
                        let _e5133_ = select(_e5077_, _e5113_, _e5132_);
                        let _e5134_ = (_e5082_ != _e5082_);
                        if _e5134_ {
                            phi_17084_ = true;
                        } else {
                            phi_17084_ = (_e5118_ <= _e5082_);
                        }
                        let _e5137_ = phi_17084_;
                        let _e5138_ = select(_e5082_, _e5118_, _e5137_);
                        if _e5119_ {
                            phi_17099_ = true;
                        } else {
                            phi_17099_ = (_e5103_ >= _e5067_);
                        }
                        let _e5141_ = phi_17099_;
                        let _e5142_ = select(_e5067_, _e5103_, _e5141_);
                        if _e5124_ {
                            phi_17114_ = true;
                        } else {
                            phi_17114_ = (_e5108_ >= _e5072_);
                        }
                        let _e5145_ = phi_17114_;
                        let _e5146_ = select(_e5072_, _e5108_, _e5145_);
                        if _e5129_ {
                            phi_17129_ = true;
                        } else {
                            phi_17129_ = (_e5113_ >= _e5077_);
                        }
                        let _e5149_ = phi_17129_;
                        let _e5150_ = select(_e5077_, _e5113_, _e5149_);
                        if _e5134_ {
                            phi_17144_ = true;
                        } else {
                            phi_17144_ = (_e5118_ >= _e5082_);
                        }
                        let _e5153_ = phi_17144_;
                        let _e5154_ = select(_e5082_, _e5118_, _e5153_);
                        if (_e5123_ != _e5123_) {
                            phi_17159_ = true;
                        } else {
                            phi_17159_ = (-1f >= _e5123_);
                        }
                        let _e5158_ = phi_17159_;
                        let _e5159_ = select(_e5123_, -1f, _e5158_);
                        if (_e5128_ != _e5128_) {
                            phi_17174_ = true;
                        } else {
                            phi_17174_ = (-1f >= _e5128_);
                        }
                        let _e5163_ = phi_17174_;
                        let _e5164_ = select(_e5128_, -1f, _e5163_);
                        if (_e5133_ != _e5133_) {
                            phi_17189_ = true;
                        } else {
                            phi_17189_ = (-1f >= _e5133_);
                        }
                        let _e5168_ = phi_17189_;
                        let _e5169_ = select(_e5133_, -1f, _e5168_);
                        if (_e5138_ != _e5138_) {
                            phi_17204_ = true;
                        } else {
                            phi_17204_ = (-1f >= _e5138_);
                        }
                        let _e5173_ = phi_17204_;
                        let _e5174_ = select(_e5138_, -1f, _e5173_);
                        if (_e5159_ != _e5159_) {
                            phi_17219_ = true;
                        } else {
                            phi_17219_ = (1f <= _e5159_);
                        }
                        let _e5178_ = phi_17219_;
                        let _e5179_ = select(_e5159_, 1f, _e5178_);
                        if (_e5164_ != _e5164_) {
                            phi_17234_ = true;
                        } else {
                            phi_17234_ = (1f <= _e5164_);
                        }
                        let _e5183_ = phi_17234_;
                        let _e5184_ = select(_e5164_, 1f, _e5183_);
                        if (_e5169_ != _e5169_) {
                            phi_17249_ = true;
                        } else {
                            phi_17249_ = (1f <= _e5169_);
                        }
                        let _e5188_ = phi_17249_;
                        let _e5189_ = select(_e5169_, 1f, _e5188_);
                        if (_e5174_ != _e5174_) {
                            phi_17264_ = true;
                        } else {
                            phi_17264_ = (1f <= _e5174_);
                        }
                        let _e5193_ = phi_17264_;
                        let _e5194_ = select(_e5174_, 1f, _e5193_);
                        if (_e5142_ != _e5142_) {
                            phi_17279_ = true;
                        } else {
                            phi_17279_ = (-1f >= _e5142_);
                        }
                        let _e5198_ = phi_17279_;
                        let _e5199_ = select(_e5142_, -1f, _e5198_);
                        if (_e5146_ != _e5146_) {
                            phi_17294_ = true;
                        } else {
                            phi_17294_ = (-1f >= _e5146_);
                        }
                        let _e5203_ = phi_17294_;
                        let _e5204_ = select(_e5146_, -1f, _e5203_);
                        if (_e5150_ != _e5150_) {
                            phi_17309_ = true;
                        } else {
                            phi_17309_ = (-1f >= _e5150_);
                        }
                        let _e5208_ = phi_17309_;
                        let _e5209_ = select(_e5150_, -1f, _e5208_);
                        if (_e5154_ != _e5154_) {
                            phi_17324_ = true;
                        } else {
                            phi_17324_ = (-1f >= _e5154_);
                        }
                        let _e5213_ = phi_17324_;
                        let _e5214_ = select(_e5154_, -1f, _e5213_);
                        if (_e5199_ != _e5199_) {
                            phi_17339_ = true;
                        } else {
                            phi_17339_ = (1f <= _e5199_);
                        }
                        let _e5218_ = phi_17339_;
                        let _e5219_ = select(_e5199_, 1f, _e5218_);
                        if (_e5204_ != _e5204_) {
                            phi_17354_ = true;
                        } else {
                            phi_17354_ = (1f <= _e5204_);
                        }
                        let _e5223_ = phi_17354_;
                        let _e5224_ = select(_e5204_, 1f, _e5223_);
                        if (_e5209_ != _e5209_) {
                            phi_17369_ = true;
                        } else {
                            phi_17369_ = (1f <= _e5209_);
                        }
                        let _e5228_ = phi_17369_;
                        let _e5229_ = select(_e5209_, 1f, _e5228_);
                        if (_e5214_ != _e5214_) {
                            phi_17384_ = true;
                        } else {
                            phi_17384_ = (1f <= _e5214_);
                        }
                        let _e5233_ = phi_17384_;
                        let _e5234_ = select(_e5214_, 1f, _e5233_);
                        let _e5239_ = (0.5f * (_e5219_ + _e5179_));
                        let _e5240_ = (0.5f * (_e5224_ + _e5184_));
                        let _e5241_ = (0.5f * (_e5229_ + _e5189_));
                        let _e5242_ = (0.5f * (_e5234_ + _e5194_));
                        let _e5243_ = (_e5219_ - _e5179_);
                        let _e5244_ = (_e5224_ - _e5184_);
                        let _e5245_ = (_e5229_ - _e5189_);
                        let _e5246_ = (_e5234_ - _e5194_);
                        let _e5247_ = (0.5f * _e5243_);
                        let _e5248_ = (0.5f * _e5244_);
                        let _e5249_ = (0.5f * _e5245_);
                        let _e5250_ = (0.5f * _e5246_);
                        let _e5251_ = (_e5239_ * _e5239_);
                        let _e5252_ = (_e5240_ * _e5240_);
                        let _e5253_ = (_e5241_ * _e5241_);
                        let _e5254_ = (_e5242_ * _e5242_);
                        let _e5255_ = (_e5247_ * _e5247_);
                        let _e5256_ = (_e5248_ * _e5248_);
                        let _e5257_ = (_e5249_ * _e5249_);
                        let _e5258_ = (_e5250_ * _e5250_);
                        if (_e5243_ != _e5243_) {
                            phi_17399_ = true;
                        } else {
                            phi_17399_ = (0f >= _e5243_);
                        }
                        let _e5262_ = phi_17399_;
                        if (_e5244_ != _e5244_) {
                            phi_17414_ = true;
                        } else {
                            phi_17414_ = (0f >= _e5244_);
                        }
                        let _e5267_ = phi_17414_;
                        if (_e5245_ != _e5245_) {
                            phi_17429_ = true;
                        } else {
                            phi_17429_ = (0f >= _e5245_);
                        }
                        let _e5272_ = phi_17429_;
                        if (_e5246_ != _e5246_) {
                            phi_17444_ = true;
                        } else {
                            phi_17444_ = (0f >= _e5246_);
                        }
                        let _e5277_ = phi_17444_;
                        let _e5279_ = (_e5142_ - _e5123_);
                        let _e5280_ = (_e5146_ - _e5128_);
                        let _e5281_ = (_e5150_ - _e5133_);
                        let _e5282_ = (_e5154_ - _e5138_);
                        let _e5285_ = bitcast<f32>((bitcast<u32>(_e5067_) & 2147483647u));
                        let _e5288_ = bitcast<f32>((bitcast<u32>(_e5072_) & 2147483647u));
                        let _e5291_ = bitcast<f32>((bitcast<u32>(_e5077_) & 2147483647u));
                        let _e5294_ = bitcast<f32>((bitcast<u32>(_e5082_) & 2147483647u));
                        let _e5297_ = bitcast<f32>((bitcast<u32>(_e5103_) & 2147483647u));
                        let _e5300_ = bitcast<f32>((bitcast<u32>(_e5108_) & 2147483647u));
                        let _e5303_ = bitcast<f32>((bitcast<u32>(_e5113_) & 2147483647u));
                        let _e5306_ = bitcast<f32>((bitcast<u32>(_e5118_) & 2147483647u));
                        if (_e5285_ != _e5285_) {
                            phi_17699_ = true;
                        } else {
                            phi_17699_ = (_e5297_ >= _e5285_);
                        }
                        let _e5310_ = phi_17699_;
                        if (_e5288_ != _e5288_) {
                            phi_17714_ = true;
                        } else {
                            phi_17714_ = (_e5300_ >= _e5288_);
                        }
                        let _e5315_ = phi_17714_;
                        if (_e5291_ != _e5291_) {
                            phi_17729_ = true;
                        } else {
                            phi_17729_ = (_e5303_ >= _e5291_);
                        }
                        let _e5320_ = phi_17729_;
                        if (_e5294_ != _e5294_) {
                            phi_17744_ = true;
                        } else {
                            phi_17744_ = (_e5306_ >= _e5294_);
                        }
                        let _e5325_ = phi_17744_;
                        let _e5331_ = (1f - (_e5251_ + _e5255_));
                        let _e5332_ = (1f - (_e5252_ + _e5256_));
                        let _e5333_ = (1f - (_e5253_ + _e5257_));
                        let _e5334_ = (1f - (_e5254_ + _e5258_));
                        let _e5360_ = select(((_e5239_ * _e5331_) * ((_e5331_ * _e5331_) + ((4f * _e5251_) * _e5255_))), 0f, (select(_e5285_, _e5297_, _e5310_) <= 0f));
                        let _e5362_ = select(((_e5240_ * _e5332_) * ((_e5332_ * _e5332_) + ((4f * _e5252_) * _e5256_))), 0f, (select(_e5288_, _e5300_, _e5315_) <= 0f));
                        let _e5364_ = select(((_e5241_ * _e5333_) * ((_e5333_ * _e5333_) + ((4f * _e5253_) * _e5257_))), 0f, (select(_e5291_, _e5303_, _e5320_) <= 0f));
                        let _e5366_ = select(((_e5242_ * _e5334_) * ((_e5334_ * _e5334_) + ((4f * _e5254_) * _e5258_))), 0f, (select(_e5294_, _e5306_, _e5325_) <= 0f));
                        if (_e5279_ != _e5279_) {
                            phi_17759_ = true;
                        } else {
                            phi_17759_ = (0.000001f >= _e5279_);
                        }
                        let _e5370_ = phi_17759_;
                        let _e5371_ = select(_e5279_, 0.000001f, _e5370_);
                        if (_e5280_ != _e5280_) {
                            phi_17774_ = true;
                        } else {
                            phi_17774_ = (0.000001f >= _e5280_);
                        }
                        let _e5375_ = phi_17774_;
                        let _e5376_ = select(_e5280_, 0.000001f, _e5375_);
                        if (_e5281_ != _e5281_) {
                            phi_17789_ = true;
                        } else {
                            phi_17789_ = (0.000001f >= _e5281_);
                        }
                        let _e5380_ = phi_17789_;
                        let _e5381_ = select(_e5281_, 0.000001f, _e5380_);
                        if (_e5282_ != _e5282_) {
                            phi_17804_ = true;
                        } else {
                            phi_17804_ = (0.000001f >= _e5282_);
                        }
                        let _e5385_ = phi_17804_;
                        let _e5386_ = select(_e5282_, 0.000001f, _e5385_);
                        let _e5389_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5371_)));
                        let _e5392_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5376_)));
                        let _e5395_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5381_)));
                        let _e5398_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5386_)));
                        let _e5408_ = (_e5389_ + (_e5389_ * (1f - (_e5371_ * _e5389_))));
                        let _e5410_ = (_e5392_ + (_e5392_ * (1f - (_e5376_ * _e5392_))));
                        let _e5412_ = (_e5395_ + (_e5395_ * (1f - (_e5381_ * _e5395_))));
                        let _e5414_ = (_e5398_ + (_e5398_ * (1f - (_e5386_ * _e5398_))));
                        let _e5491_ = (1f - _e417_);
                        let _e5492_ = (1f - _e437_);
                        let _e5493_ = (1f - _e457_);
                        let _e5494_ = (1f - _e477_);
                        phi_11111_ = type_24_(array<f32, 4>(((_e4585_.member.member[0] * _e5491_) + (_e417_ * select(select(select(_e5360_, ((select(_e5243_, 0f, _e5262_) * _e5360_) * (_e5408_ + (_e5408_ * (1f - (_e5371_ * _e5408_))))), (select((select(0f, 1f, (-1f <= _e5123_)) * select(0f, 1f, (_e5142_ <= 1f))), 1f, (_e5279_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e5123_)), 0f, (_e5142_ <= -1f)))), ((_e4585_.member.member[1] * _e5492_) + (_e437_ * select(select(select(_e5362_, ((select(_e5244_, 0f, _e5267_) * _e5362_) * (_e5410_ + (_e5410_ * (1f - (_e5376_ * _e5410_))))), (select((select(0f, 1f, (-1f <= _e5128_)) * select(0f, 1f, (_e5146_ <= 1f))), 1f, (_e5280_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e5128_)), 0f, (_e5146_ <= -1f)))), ((_e4585_.member.member[2] * _e5493_) + (_e457_ * select(select(select(_e5364_, ((select(_e5245_, 0f, _e5272_) * _e5364_) * (_e5412_ + (_e5412_ * (1f - (_e5381_ * _e5412_))))), (select((select(0f, 1f, (-1f <= _e5133_)) * select(0f, 1f, (_e5150_ <= 1f))), 1f, (_e5281_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e5133_)), 0f, (_e5150_ <= -1f)))), ((_e4585_.member.member[3] * _e5494_) + (_e477_ * select(select(select(_e5366_, ((select(_e5246_, 0f, _e5277_) * _e5366_) * (_e5414_ + (_e5414_ * (1f - (_e5386_ * _e5414_))))), (select((select(0f, 1f, (-1f <= _e5138_)) * select(0f, 1f, (_e5154_ <= 1f))), 1f, (_e5282_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e5138_)), 0f, (_e5154_ <= -1f))))));
                        phi_11112_ = type_24_(array<f32, 4>(((_e3917_.member.member[0] * _e5491_) + (_e417_ * select(select(select(_e4916_, ((select(_e4799_, 0f, _e4818_) * _e4916_) * (_e4964_ + (_e4964_ * (1f - (_e4927_ * _e4964_))))), (select((select(0f, 1f, (-1f <= _e4679_)) * select(0f, 1f, (_e4698_ <= 1f))), 1f, (_e4835_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4679_)), 0f, (_e4698_ <= -1f)))), ((_e3917_.member.member[1] * _e5492_) + (_e437_ * select(select(select(_e4918_, ((select(_e4800_, 0f, _e4823_) * _e4918_) * (_e4966_ + (_e4966_ * (1f - (_e4932_ * _e4966_))))), (select((select(0f, 1f, (-1f <= _e4684_)) * select(0f, 1f, (_e4702_ <= 1f))), 1f, (_e4836_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4684_)), 0f, (_e4702_ <= -1f)))), ((_e3917_.member.member[2] * _e5493_) + (_e457_ * select(select(select(_e4920_, ((select(_e4801_, 0f, _e4828_) * _e4920_) * (_e4968_ + (_e4968_ * (1f - (_e4937_ * _e4968_))))), (select((select(0f, 1f, (-1f <= _e4689_)) * select(0f, 1f, (_e4706_ <= 1f))), 1f, (_e4837_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4689_)), 0f, (_e4706_ <= -1f)))), ((_e3917_.member.member[3] * _e5494_) + (_e477_ * select(select(select(_e4922_, ((select(_e4802_, 0f, _e4833_) * _e4922_) * (_e4970_ + (_e4970_ * (1f - (_e4942_ * _e4970_))))), (select((select(0f, 1f, (-1f <= _e4694_)) * select(0f, 1f, (_e4710_ <= 1f))), 1f, (_e4838_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e4694_)), 0f, (_e4710_ <= -1f))))));
                    }
                    let _e5540_ = phi_11111_;
                    let _e5542_ = phi_11112_;
                    if _e948_ {
                        let _e5603_ = (_e1310_ * 6u);
                        let _e5604_ = local;
                        let _e5609_ = bitcast<u32>(_e5604_[0]);
                        let _e5610_ = (_e5609_ & 2139095040u);
                        if (_e5610_ == 2139095040u) {
                            phi_11271_ = true;
                        } else {
                            phi_11271_ = (_e5610_ == 0u);
                        }
                        let _e5614_ = phi_11271_;
                        let _e5617_ = bitcast<u32>(_e5604_[1]);
                        let _e5618_ = (_e5617_ & 2139095040u);
                        if (_e5618_ == 2139095040u) {
                            phi_11287_ = true;
                        } else {
                            phi_11287_ = (_e5618_ == 0u);
                        }
                        let _e5622_ = phi_11287_;
                        let _e5625_ = bitcast<u32>(_e5604_[2]);
                        let _e5626_ = (_e5625_ & 2139095040u);
                        if (_e5626_ == 2139095040u) {
                            phi_11303_ = true;
                        } else {
                            phi_11303_ = (_e5626_ == 0u);
                        }
                        let _e5630_ = phi_11303_;
                        let _e5633_ = bitcast<u32>(_e5604_[3]);
                        let _e5634_ = (_e5633_ & 2139095040u);
                        if (_e5634_ == 2139095040u) {
                            phi_11319_ = true;
                        } else {
                            phi_11319_ = (_e5634_ == 0u);
                        }
                        let _e5638_ = phi_11319_;
                        let _e5642_ = (_e5603_ < _e119_);
                        if _e5642_ {
                            if _e5642_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            feature_outputs.member[_e5603_] = vec4<f32>(bitcast<f32>(select(_e5609_, 0u, _e5614_)), bitcast<f32>(select(_e5617_, 0u, _e5622_)), bitcast<f32>(select(_e5625_, 0u, _e5630_)), bitcast<f32>(select(_e5633_, 0u, _e5638_)));
                        }
                        let _e5645_ = (_e5603_ + 1u);
                        let _e5646_ = local_1_;
                        let _e5651_ = bitcast<u32>(_e5646_[0]);
                        let _e5652_ = (_e5651_ & 2139095040u);
                        if (_e5652_ == 2139095040u) {
                            phi_11357_ = true;
                        } else {
                            phi_11357_ = (_e5652_ == 0u);
                        }
                        let _e5656_ = phi_11357_;
                        let _e5659_ = bitcast<u32>(_e5646_[1]);
                        let _e5660_ = (_e5659_ & 2139095040u);
                        if (_e5660_ == 2139095040u) {
                            phi_11373_ = true;
                        } else {
                            phi_11373_ = (_e5660_ == 0u);
                        }
                        let _e5664_ = phi_11373_;
                        let _e5667_ = bitcast<u32>(_e5646_[2]);
                        let _e5668_ = (_e5667_ & 2139095040u);
                        if (_e5668_ == 2139095040u) {
                            phi_11389_ = true;
                        } else {
                            phi_11389_ = (_e5668_ == 0u);
                        }
                        let _e5672_ = phi_11389_;
                        let _e5675_ = bitcast<u32>(_e5646_[3]);
                        let _e5676_ = (_e5675_ & 2139095040u);
                        if (_e5676_ == 2139095040u) {
                            phi_11405_ = true;
                        } else {
                            phi_11405_ = (_e5676_ == 0u);
                        }
                        let _e5680_ = phi_11405_;
                        let _e5684_ = (_e5645_ < _e119_);
                        if _e5684_ {
                            if _e5684_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            feature_outputs.member[_e5645_] = vec4<f32>(bitcast<f32>(select(_e5651_, 0u, _e5656_)), bitcast<f32>(select(_e5659_, 0u, _e5664_)), bitcast<f32>(select(_e5667_, 0u, _e5672_)), bitcast<f32>(select(_e5675_, 0u, _e5680_)));
                        }
                        let _e5687_ = (_e5603_ + 2u);
                        let _e5692_ = bitcast<u32>(_e5542_.member[0]);
                        let _e5693_ = (_e5692_ & 2139095040u);
                        if (_e5693_ == 2139095040u) {
                            phi_11437_ = true;
                        } else {
                            phi_11437_ = (_e5693_ == 0u);
                        }
                        let _e5697_ = phi_11437_;
                        let _e5700_ = bitcast<u32>(_e5542_.member[1]);
                        let _e5701_ = (_e5700_ & 2139095040u);
                        if (_e5701_ == 2139095040u) {
                            phi_11453_ = true;
                        } else {
                            phi_11453_ = (_e5701_ == 0u);
                        }
                        let _e5705_ = phi_11453_;
                        let _e5708_ = bitcast<u32>(_e5542_.member[2]);
                        let _e5709_ = (_e5708_ & 2139095040u);
                        if (_e5709_ == 2139095040u) {
                            phi_11469_ = true;
                        } else {
                            phi_11469_ = (_e5709_ == 0u);
                        }
                        let _e5713_ = phi_11469_;
                        let _e5716_ = bitcast<u32>(_e5542_.member[3]);
                        let _e5717_ = (_e5716_ & 2139095040u);
                        if (_e5717_ == 2139095040u) {
                            phi_11485_ = true;
                        } else {
                            phi_11485_ = (_e5717_ == 0u);
                        }
                        let _e5721_ = phi_11485_;
                        let _e5725_ = (_e5687_ < _e119_);
                        if _e5725_ {
                            if _e5725_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            feature_outputs.member[_e5687_] = vec4<f32>(bitcast<f32>(select(_e5692_, 0u, _e5697_)), bitcast<f32>(select(_e5700_, 0u, _e5705_)), bitcast<f32>(select(_e5708_, 0u, _e5713_)), bitcast<f32>(select(_e5716_, 0u, _e5721_)));
                        }
                        let _e5728_ = (_e5603_ + 3u);
                        let _e5733_ = bitcast<u32>(_e5540_.member[0]);
                        let _e5734_ = (_e5733_ & 2139095040u);
                        if (_e5734_ == 2139095040u) {
                            phi_11517_ = true;
                        } else {
                            phi_11517_ = (_e5734_ == 0u);
                        }
                        let _e5738_ = phi_11517_;
                        let _e5741_ = bitcast<u32>(_e5540_.member[1]);
                        let _e5742_ = (_e5741_ & 2139095040u);
                        if (_e5742_ == 2139095040u) {
                            phi_11533_ = true;
                        } else {
                            phi_11533_ = (_e5742_ == 0u);
                        }
                        let _e5746_ = phi_11533_;
                        let _e5749_ = bitcast<u32>(_e5540_.member[2]);
                        let _e5750_ = (_e5749_ & 2139095040u);
                        if (_e5750_ == 2139095040u) {
                            phi_11549_ = true;
                        } else {
                            phi_11549_ = (_e5750_ == 0u);
                        }
                        let _e5754_ = phi_11549_;
                        let _e5757_ = bitcast<u32>(_e5540_.member[3]);
                        let _e5758_ = (_e5757_ & 2139095040u);
                        if (_e5758_ == 2139095040u) {
                            phi_11565_ = true;
                        } else {
                            phi_11565_ = (_e5758_ == 0u);
                        }
                        let _e5762_ = phi_11565_;
                        let _e5766_ = (_e5728_ < _e119_);
                        if _e5766_ {
                            if _e5766_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            feature_outputs.member[_e5728_] = vec4<f32>(bitcast<f32>(select(_e5733_, 0u, _e5738_)), bitcast<f32>(select(_e5741_, 0u, _e5746_)), bitcast<f32>(select(_e5749_, 0u, _e5754_)), bitcast<f32>(select(_e5757_, 0u, _e5762_)));
                        }
                        let _e5769_ = (_e5603_ + 4u);
                        let _e5770_ = bitcast<u32>((_e3917_.member_1_.member[0] + (_e388_.x * (_e5542_.member[0] * _e5542_.member[1]))));
                        let _e5771_ = (_e5770_ & 2139095040u);
                        if (_e5771_ == 2139095040u) {
                            phi_11597_ = true;
                        } else {
                            phi_11597_ = (_e5771_ == 0u);
                        }
                        let _e5775_ = phi_11597_;
                        let _e5778_ = bitcast<u32>((_e3917_.member_1_.member[1] + (_e388_.y * (_e5542_.member[1] * _e5542_.member[2]))));
                        let _e5779_ = (_e5778_ & 2139095040u);
                        if (_e5779_ == 2139095040u) {
                            phi_11613_ = true;
                        } else {
                            phi_11613_ = (_e5779_ == 0u);
                        }
                        let _e5783_ = phi_11613_;
                        let _e5786_ = bitcast<u32>((_e3917_.member_1_.member[2] + (_e388_.z * (_e5542_.member[2] * _e5542_.member[3]))));
                        let _e5787_ = (_e5786_ & 2139095040u);
                        if (_e5787_ == 2139095040u) {
                            phi_11629_ = true;
                        } else {
                            phi_11629_ = (_e5787_ == 0u);
                        }
                        let _e5791_ = phi_11629_;
                        let _e5794_ = bitcast<u32>((_e3917_.member_1_.member[3] + (_e388_.w * (_e5542_.member[3] * _e5542_.member[0]))));
                        let _e5795_ = (_e5794_ & 2139095040u);
                        if (_e5795_ == 2139095040u) {
                            phi_11645_ = true;
                        } else {
                            phi_11645_ = (_e5795_ == 0u);
                        }
                        let _e5799_ = phi_11645_;
                        let _e5803_ = (_e5769_ < _e119_);
                        if _e5803_ {
                            if _e5803_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            feature_outputs.member[_e5769_] = vec4<f32>(bitcast<f32>(select(_e5770_, 0u, _e5775_)), bitcast<f32>(select(_e5778_, 0u, _e5783_)), bitcast<f32>(select(_e5786_, 0u, _e5791_)), bitcast<f32>(select(_e5794_, 0u, _e5799_)));
                        }
                        let _e5806_ = (_e5603_ + 5u);
                        let _e5807_ = bitcast<u32>((_e4585_.member_1_.member[0] + (_e388_.x * (_e5540_.member[0] * _e5540_.member[1]))));
                        let _e5808_ = (_e5807_ & 2139095040u);
                        if (_e5808_ == 2139095040u) {
                            phi_11677_ = true;
                        } else {
                            phi_11677_ = (_e5808_ == 0u);
                        }
                        let _e5812_ = phi_11677_;
                        let _e5815_ = bitcast<u32>((_e4585_.member_1_.member[1] + (_e388_.y * (_e5540_.member[1] * _e5540_.member[2]))));
                        let _e5816_ = (_e5815_ & 2139095040u);
                        if (_e5816_ == 2139095040u) {
                            phi_11693_ = true;
                        } else {
                            phi_11693_ = (_e5816_ == 0u);
                        }
                        let _e5820_ = phi_11693_;
                        let _e5823_ = bitcast<u32>((_e4585_.member_1_.member[2] + (_e388_.z * (_e5540_.member[2] * _e5540_.member[3]))));
                        let _e5824_ = (_e5823_ & 2139095040u);
                        if (_e5824_ == 2139095040u) {
                            phi_11709_ = true;
                        } else {
                            phi_11709_ = (_e5824_ == 0u);
                        }
                        let _e5828_ = phi_11709_;
                        let _e5831_ = bitcast<u32>((_e4585_.member_1_.member[3] + (_e388_.w * (_e5540_.member[3] * _e5540_.member[0]))));
                        let _e5832_ = (_e5831_ & 2139095040u);
                        if (_e5832_ == 2139095040u) {
                            phi_11725_ = true;
                        } else {
                            phi_11725_ = (_e5832_ == 0u);
                        }
                        let _e5836_ = phi_11725_;
                        let _e5840_ = (_e5806_ < _e119_);
                        if _e5840_ {
                            if _e5840_ {
                            } else {
                                phi_18515_ = true;
                                break;
                            }
                            feature_outputs.member[_e5806_] = vec4<f32>(bitcast<f32>(select(_e5807_, 0u, _e5812_)), bitcast<f32>(select(_e5815_, 0u, _e5820_)), bitcast<f32>(select(_e5823_, 0u, _e5828_)), bitcast<f32>(select(_e5831_, 0u, _e5836_)));
                        }
                    }
                    if (_e946_ > 0u) {
                        let _e5845_ = ((_e946_ - 1u) & 255u);
                        if (_e5845_ < 512u) {
                        } else {
                            phi_18515_ = true;
                            break;
                        }
                        let _e5848_ = lane_buf[_e5845_];
                        let _e5850_ = bitcast<u32>(_e5848_.x);
                        let _e5851_ = (_e5850_ & 2139095040u);
                        if (_e5851_ == 2139095040u) {
                            phi_11766_ = true;
                        } else {
                            phi_11766_ = (_e5851_ == 0u);
                        }
                        let _e5855_ = phi_11766_;
                        let _e5857_ = bitcast<f32>(select(_e5850_, 0u, _e5855_));
                        if (_e5857_ < -1000000000000000000f) {
                            phi_11783_ = -1000000000000000000f;
                        } else {
                            phi_11783_ = select(_e5857_, 1000000000000000000f, (_e5857_ > 1000000000000000000f));
                        }
                        let _e5862_ = phi_11783_;
                        let _e5864_ = bitcast<u32>(_e5848_.y);
                        let _e5865_ = (_e5864_ & 2139095040u);
                        if (_e5865_ == 2139095040u) {
                            phi_11795_ = true;
                        } else {
                            phi_11795_ = (_e5865_ == 0u);
                        }
                        let _e5869_ = phi_11795_;
                        let _e5871_ = bitcast<f32>(select(_e5864_, 0u, _e5869_));
                        if (_e5871_ < -1000000000000000000f) {
                            phi_11812_ = -1000000000000000000f;
                        } else {
                            phi_11812_ = select(_e5871_, 1000000000000000000f, (_e5871_ > 1000000000000000000f));
                        }
                        let _e5876_ = phi_11812_;
                        let _e5878_ = bitcast<u32>(_e5848_.z);
                        let _e5879_ = (_e5878_ & 2139095040u);
                        if (_e5879_ == 2139095040u) {
                            phi_11824_ = true;
                        } else {
                            phi_11824_ = (_e5879_ == 0u);
                        }
                        let _e5883_ = phi_11824_;
                        let _e5885_ = bitcast<f32>(select(_e5878_, 0u, _e5883_));
                        if (_e5885_ < -1000000000000000000f) {
                            phi_11841_ = -1000000000000000000f;
                        } else {
                            phi_11841_ = select(_e5885_, 1000000000000000000f, (_e5885_ > 1000000000000000000f));
                        }
                        let _e5890_ = phi_11841_;
                        let _e5892_ = bitcast<u32>(_e5848_.w);
                        let _e5893_ = (_e5892_ & 2139095040u);
                        if (_e5893_ == 2139095040u) {
                            phi_11853_ = true;
                        } else {
                            phi_11853_ = (_e5893_ == 0u);
                        }
                        let _e5897_ = phi_11853_;
                        let _e5899_ = bitcast<f32>(select(_e5892_, 0u, _e5897_));
                        if (_e5899_ < -1000000000000000000f) {
                            phi_11870_ = -1000000000000000000f;
                        } else {
                            phi_11870_ = select(_e5899_, 1000000000000000000f, (_e5899_ > 1000000000000000000f));
                        }
                        let _e5904_ = phi_11870_;
                        let _e5906_ = (256u + _e5845_);
                        if (_e5906_ < 512u) {
                        } else {
                            phi_18515_ = true;
                            break;
                        }
                        let _e5909_ = lane_buf[_e5906_];
                        let _e5911_ = bitcast<u32>(_e5909_.x);
                        let _e5912_ = (_e5911_ & 2139095040u);
                        if (_e5912_ == 2139095040u) {
                            phi_11891_ = true;
                        } else {
                            phi_11891_ = (_e5912_ == 0u);
                        }
                        let _e5916_ = phi_11891_;
                        let _e5918_ = bitcast<f32>(select(_e5911_, 0u, _e5916_));
                        if (_e5918_ < -1000000000000000000f) {
                            phi_11908_ = -1000000000000000000f;
                        } else {
                            phi_11908_ = select(_e5918_, 1000000000000000000f, (_e5918_ > 1000000000000000000f));
                        }
                        let _e5923_ = phi_11908_;
                        let _e5925_ = bitcast<u32>(_e5909_.y);
                        let _e5926_ = (_e5925_ & 2139095040u);
                        if (_e5926_ == 2139095040u) {
                            phi_11920_ = true;
                        } else {
                            phi_11920_ = (_e5926_ == 0u);
                        }
                        let _e5930_ = phi_11920_;
                        let _e5932_ = bitcast<f32>(select(_e5925_, 0u, _e5930_));
                        if (_e5932_ < -1000000000000000000f) {
                            phi_11937_ = -1000000000000000000f;
                        } else {
                            phi_11937_ = select(_e5932_, 1000000000000000000f, (_e5932_ > 1000000000000000000f));
                        }
                        let _e5937_ = phi_11937_;
                        let _e5939_ = bitcast<u32>(_e5909_.z);
                        let _e5940_ = (_e5939_ & 2139095040u);
                        if (_e5940_ == 2139095040u) {
                            phi_11949_ = true;
                        } else {
                            phi_11949_ = (_e5940_ == 0u);
                        }
                        let _e5944_ = phi_11949_;
                        let _e5946_ = bitcast<f32>(select(_e5939_, 0u, _e5944_));
                        if (_e5946_ < -1000000000000000000f) {
                            phi_11966_ = -1000000000000000000f;
                        } else {
                            phi_11966_ = select(_e5946_, 1000000000000000000f, (_e5946_ > 1000000000000000000f));
                        }
                        let _e5951_ = phi_11966_;
                        let _e5953_ = bitcast<u32>(_e5909_.w);
                        let _e5954_ = (_e5953_ & 2139095040u);
                        if (_e5954_ == 2139095040u) {
                            phi_11978_ = true;
                        } else {
                            phi_11978_ = (_e5954_ == 0u);
                        }
                        let _e5958_ = phi_11978_;
                        let _e5960_ = bitcast<f32>(select(_e5953_, 0u, _e5958_));
                        if (_e5960_ < -1000000000000000000f) {
                            phi_11995_ = -1000000000000000000f;
                        } else {
                            phi_11995_ = select(_e5960_, 1000000000000000000f, (_e5960_ > 1000000000000000000f));
                        }
                        let _e5965_ = phi_11995_;
                        phi_12000_ = type_32_(_e3210_.member, _e3210_.member_1_, _e3210_.member_2_, type_27_(type_24_(array<f32, 4>(_e5862_, _e5876_, _e5890_, _e5904_)), type_24_(array<f32, 4>(_e5923_, _e5937_, _e5951_, _e5965_))));
                    } else {
                        phi_12000_ = _e3210_;
                    }
                    let _e5976_ = phi_12000_;
                    workgroupBarrier();
                    phi_18516_ = _e3008_;
                    phi_1644_ = _e5976_;
                } else {
                    phi_18516_ = _e916_;
                    phi_1644_ = type_32_();
                }
                let _e5978_ = phi_18516_;
                let _e5980_ = phi_1644_;
                phi_17856_ = _e5978_;
                phi_1640_ = _e933_;
                phi_1643_ = _e5980_;
                phi_18515_ = _e5978_;
                if !(_e939_) {
                    break;
                }
            }
            let _e5983_ = phi_18515_;
            if _e5983_ {
                break;
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn fused_core_scan_256(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(num_workgroups) num_wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    num_wg_1_ = num_wg;
    lid_1_ = lid;
    function_();
    return;
}
