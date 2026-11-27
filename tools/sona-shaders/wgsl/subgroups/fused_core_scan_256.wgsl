struct type_7_ {
    member: array<vec4<f32>>,
}

struct type_15_ {
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

struct type_29_ {
    member: f32,
    member_1_: f32,
}

struct type_30_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_32_ {
    member: array<f32, 4>,
}

struct type_33_ {
    member: type_32_,
    member_1_: type_32_,
    member_2_: type_32_,
}

struct type_34_ {
    member: i32,
    member_1_: i32,
}

struct type_35_ {
    member: u32,
    member_1_: i32,
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
var<workgroup> aux_scratch: array<f32, 256>;
var<workgroup> lane_buf: array<f32, 2048>;

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
    var local_12_: array<array<u32, 4>, 4>;
    var local_13_: array<f32, 4>;
    var local_14_: array<f32, 4>;
    var local_15_: array<f32, 4>;
    var local_16_: array<f32, 4>;
    var local_17_: array<f32, 4>;
    var local_18_: array<f32, 4>;
    var local_19_: array<f32, 4>;
    var local_20_: array<f32, 4>;
    var local_21_: array<f32, 4>;
    var local_22_: array<f32, 4>;
    var phi_303_: u32;
    var phi_304_: u32;
    var phi_321_: bool;
    var phi_329_: bool;
    var phi_339_: u32;
    var phi_370_: u32;
    var phi_381_: u32;
    var phi_384_: u32;
    var phi_382_: u32;
    var phi_385_: u32;
    var phi_396_: u32;
    var phi_399_: u32;
    var phi_401_: u32;
    var phi_403_: u32;
    var phi_417_: u32;
    var phi_418_: u32;
    var phi_397_: u32;
    var phi_400_: u32;
    var phi_402_: u32;
    var phi_404_: u32;
    var local_23_: u32;
    var phi_422_: u32;
    var phi_423_: u32;
    var phi_437_: u32;
    var phi_438_: u32;
    var phi_452_: vec4<f32>;
    var phi_465_: vec4<f32>;
    var phi_478_: vec4<f32>;
    var phi_491_: vec4<f32>;
    var phi_504_: vec4<f32>;
    var phi_517_: vec4<f32>;
    var phi_530_: vec4<f32>;
    var phi_543_: vec4<f32>;
    var phi_556_: vec4<f32>;
    var phi_569_: vec4<f32>;
    var phi_582_: vec4<f32>;
    var phi_595_: vec4<f32>;
    var phi_608_: vec4<f32>;
    var phi_621_: vec4<f32>;
    var phi_634_: vec4<f32>;
    var phi_647_: vec4<f32>;
    var phi_660_: vec4<f32>;
    var phi_673_: vec4<f32>;
    var phi_686_: vec4<f32>;
    var phi_699_: vec4<f32>;
    var phi_712_: vec4<f32>;
    var phi_725_: vec4<f32>;
    var phi_738_: vec4<f32>;
    var phi_751_: vec4<f32>;
    var phi_764_: vec4<f32>;
    var phi_777_: vec4<f32>;
    var phi_790_: vec4<f32>;
    var phi_803_: vec4<f32>;
    var phi_830_: bool;
    var phi_836_: f32;
    var phi_837_: f32;
    var phi_844_: f32;
    var phi_870_: bool;
    var phi_876_: f32;
    var phi_877_: f32;
    var phi_884_: f32;
    var phi_910_: bool;
    var phi_916_: f32;
    var phi_917_: f32;
    var phi_924_: f32;
    var phi_950_: bool;
    var phi_956_: f32;
    var phi_957_: f32;
    var phi_964_: f32;
    var phi_1043_: bool;
    var phi_1049_: f32;
    var phi_1050_: f32;
    var phi_1051_: f32;
    var phi_1109_: bool;
    var phi_1115_: f32;
    var phi_1116_: f32;
    var phi_1117_: f32;
    var phi_1175_: bool;
    var phi_1181_: f32;
    var phi_1182_: f32;
    var phi_1183_: f32;
    var phi_1241_: bool;
    var phi_1247_: f32;
    var phi_1248_: f32;
    var phi_1249_: f32;
    var phi_1308_: bool;
    var phi_1314_: f32;
    var phi_1315_: f32;
    var phi_1316_: f32;
    var phi_1374_: bool;
    var phi_1380_: f32;
    var phi_1381_: f32;
    var phi_1382_: f32;
    var phi_1440_: bool;
    var phi_1446_: f32;
    var phi_1447_: f32;
    var phi_1448_: f32;
    var phi_1506_: bool;
    var phi_1512_: f32;
    var phi_1513_: f32;
    var phi_1514_: f32;
    var phi_1572_: bool;
    var phi_1578_: f32;
    var phi_1579_: f32;
    var phi_1580_: f32;
    var phi_1638_: bool;
    var phi_1644_: f32;
    var phi_1645_: f32;
    var phi_1646_: f32;
    var phi_1704_: bool;
    var phi_1710_: f32;
    var phi_1711_: f32;
    var phi_1712_: f32;
    var phi_1770_: bool;
    var phi_1776_: f32;
    var phi_1777_: f32;
    var phi_1778_: f32;
    var phi_1836_: bool;
    var phi_1842_: f32;
    var phi_1843_: f32;
    var phi_1844_: f32;
    var phi_1902_: bool;
    var phi_1908_: f32;
    var phi_1909_: f32;
    var phi_1910_: f32;
    var phi_1968_: bool;
    var phi_1974_: f32;
    var phi_1975_: f32;
    var phi_1976_: f32;
    var phi_2034_: bool;
    var phi_2040_: f32;
    var phi_2041_: f32;
    var phi_2042_: f32;
    var phi_2101_: bool;
    var phi_2107_: f32;
    var phi_2108_: f32;
    var phi_2109_: f32;
    var phi_2167_: bool;
    var phi_2173_: f32;
    var phi_2174_: f32;
    var phi_2175_: f32;
    var phi_2233_: bool;
    var phi_2239_: f32;
    var phi_2240_: f32;
    var phi_2241_: f32;
    var phi_2299_: bool;
    var phi_2305_: f32;
    var phi_2306_: f32;
    var phi_2307_: f32;
    var phi_2366_: bool;
    var phi_2372_: f32;
    var phi_2373_: f32;
    var phi_2374_: f32;
    var phi_2432_: bool;
    var phi_2438_: f32;
    var phi_2439_: f32;
    var phi_2440_: f32;
    var phi_2498_: bool;
    var phi_2504_: f32;
    var phi_2505_: f32;
    var phi_2506_: f32;
    var phi_2564_: bool;
    var phi_2570_: f32;
    var phi_2571_: f32;
    var phi_2572_: f32;
    var phi_2607_: vec4<f32>;
    var phi_2619_: bool;
    var phi_2636_: bool;
    var phi_2653_: bool;
    var phi_2670_: bool;
    var phi_2694_: vec4<f32>;
    var phi_2706_: bool;
    var phi_2723_: bool;
    var phi_2740_: bool;
    var phi_2757_: bool;
    var phi_2781_: vec4<f32>;
    var phi_2793_: bool;
    var phi_2810_: bool;
    var phi_2827_: bool;
    var phi_2844_: bool;
    var phi_2868_: vec4<f32>;
    var phi_2880_: bool;
    var phi_2897_: bool;
    var phi_2914_: bool;
    var phi_2931_: bool;
    var phi_2955_: vec4<f32>;
    var phi_2967_: bool;
    var phi_2984_: bool;
    var phi_3001_: bool;
    var phi_3018_: bool;
    var phi_3042_: vec4<f32>;
    var phi_3054_: bool;
    var phi_3071_: bool;
    var phi_3088_: bool;
    var phi_3105_: bool;
    var phi_3129_: vec4<f32>;
    var phi_23974_: bool;
    var phi_24034_: bool;
    var phi_24094_: bool;
    var phi_24154_: bool;
    var phi_24214_: bool;
    var phi_24274_: bool;
    var phi_24334_: bool;
    var phi_24394_: bool;
    var phi_3320_: vec4<f32>;
    var phi_24454_: bool;
    var phi_24514_: bool;
    var phi_24574_: bool;
    var phi_24634_: bool;
    var phi_24694_: bool;
    var phi_24754_: bool;
    var phi_24814_: bool;
    var phi_24874_: bool;
    var phi_3511_: vec4<f32>;
    var phi_3523_: bool;
    var phi_3540_: bool;
    var phi_3557_: bool;
    var phi_3574_: bool;
    var phi_3598_: vec4<f32>;
    var phi_3610_: bool;
    var phi_3627_: bool;
    var phi_3644_: bool;
    var phi_3661_: bool;
    var phi_3685_: vec4<f32>;
    var phi_3697_: bool;
    var phi_3714_: bool;
    var phi_3731_: bool;
    var phi_3748_: bool;
    var phi_3772_: vec4<f32>;
    var phi_3784_: bool;
    var phi_3801_: bool;
    var phi_3818_: bool;
    var phi_3835_: bool;
    var phi_3849_: type_15_;
    var phi_3850_: type_15_;
    var phi_3873_: type_15_;
    var phi_35544_: bool;
    var phi_35979_: bool;
    var phi_35978_: bool;
    var phi_4049_: bool;
    var phi_35977_: bool;
    var phi_4051_: bool;
    var phi_4055_: bool;
    var phi_4056_: bool;
    var phi_35976_: bool;
    var phi_4057_: bool;
    var phi_4060_: bool;
    var phi_4061_: bool;
    var phi_4062_: bool;
    var phi_4072_: bool;
    var phi_4079_: type_15_;
    var phi_4080_: type_15_;
    var phi_4103_: type_15_;
    var phi_35971_: bool;
    var phi_36012_: bool;
    var phi_36009_: bool;
    var phi_4129_: type_15_;
    var phi_4130_: type_15_;
    var phi_4153_: type_15_;
    var phi_4174_: bool;
    var phi_4222_: bool;
    var phi_4238_: bool;
    var phi_4254_: bool;
    var phi_4270_: bool;
    var phi_4329_: bool;
    var phi_4345_: bool;
    var phi_4361_: bool;
    var phi_4377_: bool;
    var phi_4436_: bool;
    var phi_4452_: bool;
    var phi_4468_: bool;
    var phi_4484_: bool;
    var phi_4543_: bool;
    var phi_4559_: bool;
    var phi_4575_: bool;
    var phi_4591_: bool;
    var phi_4650_: bool;
    var phi_4666_: bool;
    var phi_4682_: bool;
    var phi_4698_: bool;
    var phi_4757_: bool;
    var phi_4773_: bool;
    var phi_4789_: bool;
    var phi_4805_: bool;
    var phi_4864_: bool;
    var phi_4880_: bool;
    var phi_4896_: bool;
    var phi_4912_: bool;
    var phi_4971_: bool;
    var phi_4987_: bool;
    var phi_5003_: bool;
    var phi_5019_: bool;
    var phi_5078_: bool;
    var phi_5094_: bool;
    var phi_5110_: bool;
    var phi_5126_: bool;
    var phi_5185_: bool;
    var phi_5201_: bool;
    var phi_5217_: bool;
    var phi_5233_: bool;
    var phi_5292_: bool;
    var phi_5308_: bool;
    var phi_5324_: bool;
    var phi_5340_: bool;
    var phi_5399_: bool;
    var phi_5415_: bool;
    var phi_5431_: bool;
    var phi_5447_: bool;
    var phi_36006_: bool;
    var phi_5466_: type_15_;
    var phi_5467_: type_15_;
    var phi_5490_: type_15_;
    var phi_5505_: u32;
    var phi_5534_: vec4<f32>;
    var phi_5547_: vec4<f32>;
    var phi_5548_: vec4<f32>;
    var phi_5549_: vec4<f32>;
    var phi_36001_: bool;
    var phi_5552_: type_15_;
    var phi_5553_: type_15_;
    var phi_5576_: type_15_;
    var phi_5594_: f32;
    var phi_5605_: f32;
    var phi_5781_: f32;
    var phi_5792_: f32;
    var phi_5803_: f32;
    var phi_5815_: f32;
    var phi_5826_: f32;
    var phi_5837_: f32;
    var phi_5848_: f32;
    var phi_6063_: f32;
    var phi_6581_: f32;
    var phi_6592_: bool;
    var phi_6598_: f32;
    var phi_6603_: u32;
    var phi_6609_: u32;
    var phi_7718_: f32;
    var phi_7723_: f32;
    var phi_7740_: f32;
    var phi_7741_: f32;
    var phi_7742_: f32;
    var phi_7743_: f32;
    var phi_7760_: f32;
    var phi_7763_: f32;
    var phi_7765_: f32;
    var phi_7767_: f32;
    var phi_7769_: u32;
    var phi_7771_: f32;
    var phi_7773_: f32;
    var phi_7812_: f32;
    var phi_7813_: f32;
    var phi_7761_: f32;
    var phi_7764_: f32;
    var phi_7766_: f32;
    var phi_7768_: f32;
    var phi_7770_: u32;
    var phi_7772_: f32;
    var phi_7774_: f32;
    var phi_35980_: bool;
    var phi_7865_: f32;
    var phi_7866_: f32;
    var phi_7876_: u32;
    var phi_7895_: array<f32, 2>;
    var phi_8139_: bool;
    var phi_8162_: bool;
    var phi_8190_: bool;
    var phi_8214_: bool;
    var phi_6618_: u32;
    var phi_6624_: u32;
    var phi_24934_: bool;
    var phi_24994_: bool;
    var phi_6759_: f32;
    var phi_6764_: f32;
    var phi_6781_: f32;
    var phi_6782_: f32;
    var phi_6783_: f32;
    var phi_6784_: f32;
    var phi_6801_: f32;
    var phi_6804_: f32;
    var phi_6806_: f32;
    var phi_6808_: f32;
    var phi_6810_: u32;
    var phi_6812_: f32;
    var phi_6814_: f32;
    var phi_6853_: f32;
    var phi_6854_: f32;
    var phi_6802_: f32;
    var phi_6805_: f32;
    var phi_6807_: f32;
    var phi_6809_: f32;
    var phi_6811_: u32;
    var phi_6813_: f32;
    var phi_6815_: f32;
    var phi_36132_: bool;
    var phi_6906_: f32;
    var phi_6907_: f32;
    var phi_6917_: u32;
    var phi_6936_: array<f32, 2>;
    var phi_6993_: type_30_;
    var phi_7071_: u32;
    var phi_7074_: type_30_;
    var phi_7176_: type_30_;
    var phi_7072_: u32;
    var phi_7075_: type_30_;
    var phi_36161_: bool;
    var phi_7300_: array<f32, 2>;
    var phi_7312_: u32;
    var phi_7376_: array<f32, 2>;
    var phi_7458_: bool;
    var phi_7481_: bool;
    var phi_7505_: bool;
    var phi_7529_: bool;
    var phi_7557_: bool;
    var phi_7581_: bool;
    var phi_7605_: bool;
    var phi_7629_: bool;
    var phi_36176_: bool;
    var phi_36175_: bool;
    var phi_36174_: bool;
    var phi_36241_: bool;
    var phi_8229_: type_15_;
    var phi_8230_: type_15_;
    var phi_8253_: type_15_;
    var phi_8277_: f32;
    var phi_8288_: f32;
    var phi_8299_: f32;
    var phi_8310_: f32;
    var phi_8321_: f32;
    var phi_8332_: f32;
    var phi_8509_: bool;
    var phi_8515_: f32;
    var phi_8516_: f32;
    var phi_8523_: f32;
    var phi_8548_: bool;
    var phi_8554_: f32;
    var phi_8555_: f32;
    var phi_8562_: f32;
    var phi_8587_: bool;
    var phi_8593_: f32;
    var phi_8594_: f32;
    var phi_8601_: f32;
    var phi_8626_: bool;
    var phi_8632_: f32;
    var phi_8633_: f32;
    var phi_8640_: f32;
    var phi_8665_: bool;
    var phi_8671_: f32;
    var phi_8672_: f32;
    var phi_8679_: f32;
    var phi_8781_: f32;
    var phi_9281_: bool;
    var phi_9328_: bool;
    var phi_9345_: bool;
    var phi_25054_: bool;
    var phi_9358_: f32;
    var phi_9375_: f32;
    var phi_9494_: bool;
    var phi_9500_: f32;
    var phi_9501_: f32;
    var phi_9420_: f32;
    var phi_9422_: f32;
    var phi_9423_: bool;
    var phi_9502_: f32;
    var phi_9513_: bool;
    var phi_25069_: bool;
    var phi_25129_: bool;
    var phi_25189_: bool;
    var phi_9737_: bool;
    var phi_9743_: f32;
    var phi_9744_: f32;
    var phi_9663_: f32;
    var phi_9665_: f32;
    var phi_9666_: bool;
    var phi_9686_: f32;
    var phi_9745_: f32;
    var phi_9758_: f32;
    var phi_9772_: bool;
    var phi_9789_: bool;
    var phi_9797_: type_29_;
    var phi_9838_: u32;
    var phi_9841_: type_29_;
    var phi_9881_: type_29_;
    var phi_9839_: u32;
    var phi_9842_: type_29_;
    var phi_36215_: bool;
    var local_24_: type_29_;
    var local_25_: type_29_;
    var phi_9917_: u32;
    var phi_9938_: f32;
    var phi_9952_: bool;
    var phi_9983_: bool;
    var phi_10010_: bool;
    var phi_36344_: bool;
    var phi_36343_: bool;
    var phi_36342_: bool;
    var phi_16281_: f32;
    var phi_16292_: f32;
    var phi_16303_: f32;
    var phi_16314_: f32;
    var phi_16325_: f32;
    var phi_16336_: f32;
    var phi_16513_: bool;
    var phi_16519_: f32;
    var phi_16520_: f32;
    var phi_16527_: f32;
    var phi_16552_: bool;
    var phi_16558_: f32;
    var phi_16559_: f32;
    var phi_16566_: f32;
    var phi_16591_: bool;
    var phi_16597_: f32;
    var phi_16598_: f32;
    var phi_16605_: f32;
    var phi_16630_: bool;
    var phi_16636_: f32;
    var phi_16637_: f32;
    var phi_16644_: f32;
    var phi_16669_: bool;
    var phi_16675_: f32;
    var phi_16676_: f32;
    var phi_16683_: f32;
    var phi_16800_: f32;
    var phi_17300_: bool;
    var phi_17326_: u32;
    var phi_17336_: type_15_;
    var phi_17339_: f32;
    var phi_17337_: type_15_;
    var phi_17362_: type_15_;
    var phi_17407_: bool;
    var phi_25249_: bool;
    var phi_25309_: bool;
    var phi_25369_: bool;
    var phi_25429_: bool;
    var phi_25489_: bool;
    var phi_25549_: bool;
    var phi_25609_: bool;
    var phi_25669_: bool;
    var phi_17856_: bool;
    var phi_17340_: f32;
    var phi_36356_: bool;
    var local_26_: f32;
    var phi_17914_: bool;
    var phi_37001_: bool;
    var phi_37000_: bool;
    var phi_36489_: bool;
    var phi_10028_: type_15_;
    var phi_10029_: type_15_;
    var phi_10052_: type_15_;
    var phi_10076_: f32;
    var phi_10087_: f32;
    var phi_10098_: f32;
    var phi_10109_: f32;
    var phi_10120_: f32;
    var phi_10131_: f32;
    var phi_10308_: bool;
    var phi_10314_: f32;
    var phi_10315_: f32;
    var phi_10322_: f32;
    var phi_10347_: bool;
    var phi_10353_: f32;
    var phi_10354_: f32;
    var phi_10361_: f32;
    var phi_10386_: bool;
    var phi_10392_: f32;
    var phi_10393_: f32;
    var phi_10400_: f32;
    var phi_10425_: bool;
    var phi_10431_: f32;
    var phi_10432_: f32;
    var phi_10439_: f32;
    var phi_10464_: bool;
    var phi_10470_: f32;
    var phi_10471_: f32;
    var phi_10478_: f32;
    var phi_10595_: f32;
    var phi_11095_: bool;
    var phi_11122_: f32;
    var phi_11142_: f32;
    var phi_36461_: bool;
    var phi_11173_: type_15_;
    var phi_11176_: f32;
    var phi_11178_: bool;
    var phi_11174_: type_15_;
    var phi_11201_: type_15_;
    var phi_11224_: f32;
    var phi_11252_: bool;
    var phi_25729_: bool;
    var phi_25789_: bool;
    var phi_25849_: bool;
    var phi_25909_: bool;
    var phi_25969_: bool;
    var phi_26029_: bool;
    var phi_26089_: bool;
    var phi_26149_: bool;
    var phi_11701_: bool;
    var phi_11749_: bool;
    var phi_11765_: bool;
    var phi_11781_: bool;
    var phi_11798_: bool;
    var phi_26209_: bool;
    var phi_11811_: f32;
    var phi_11828_: f32;
    var phi_11947_: bool;
    var phi_11953_: f32;
    var phi_11954_: f32;
    var phi_11873_: f32;
    var phi_11875_: f32;
    var phi_11876_: bool;
    var phi_11955_: f32;
    var phi_26224_: bool;
    var phi_26284_: bool;
    var phi_12159_: f32;
    var phi_12167_: f32;
    var phi_12296_: bool;
    var phi_12302_: f32;
    var phi_12303_: f32;
    var phi_12222_: f32;
    var phi_12224_: f32;
    var phi_12225_: bool;
    var phi_12245_: f32;
    var phi_12304_: f32;
    var phi_12307_: type_29_;
    var phi_26404_: bool;
    var phi_12424_: bool;
    var phi_26464_: bool;
    var phi_12389_: bool;
    var phi_12430_: f32;
    var phi_12431_: f32;
    var phi_12432_: f32;
    var phi_12433_: bool;
    var phi_12439_: f32;
    var phi_12441_: f32;
    var phi_12442_: bool;
    var phi_12447_: f32;
    var phi_12485_: u32;
    var phi_12486_: u32;
    var phi_36452_: bool;
    var phi_12545_: type_29_;
    var phi_12573_: u32;
    var phi_12576_: type_29_;
    var phi_12616_: type_29_;
    var phi_12574_: u32;
    var phi_12577_: type_29_;
    var phi_36661_: bool;
    var local_27_: type_29_;
    var local_28_: type_29_;
    var phi_12652_: u32;
    var phi_36676_: bool;
    var phi_12674_: type_34_;
    var phi_12677_: f32;
    var phi_12679_: f32;
    var phi_12681_: bool;
    var phi_12683_: bool;
    var phi_12710_: type_35_;
    var phi_12675_: type_34_;
    var phi_12724_: type_35_;
    var phi_12856_: bool;
    var phi_12862_: f32;
    var phi_12863_: f32;
    var phi_12782_: f32;
    var phi_12784_: f32;
    var phi_12785_: bool;
    var phi_12805_: f32;
    var phi_12864_: f32;
    var phi_12865_: f32;
    var phi_12886_: f32;
    var phi_12914_: bool;
    var phi_26524_: bool;
    var phi_26584_: bool;
    var phi_26644_: bool;
    var phi_26704_: bool;
    var phi_26764_: bool;
    var phi_26824_: bool;
    var phi_26884_: bool;
    var phi_26944_: bool;
    var phi_13363_: bool;
    var phi_27199_: bool;
    var phi_14042_: bool;
    var phi_27259_: bool;
    var phi_14007_: bool;
    var phi_14048_: f32;
    var phi_14049_: f32;
    var phi_14050_: f32;
    var phi_14051_: bool;
    var phi_14057_: f32;
    var phi_14059_: f32;
    var phi_14060_: bool;
    var phi_14065_: f32;
    var phi_14103_: u32;
    var phi_14104_: u32;
    var phi_36667_: bool;
    var phi_14160_: bool;
    var phi_14178_: bool;
    var phi_36768_: bool;
    var phi_14180_: f32;
    var phi_14181_: f32;
    var phi_14182_: bool;
    var phi_14183_: bool;
    var phi_36767_: bool;
    var phi_12678_: f32;
    var phi_12680_: f32;
    var phi_12682_: bool;
    var phi_12684_: bool;
    var phi_36766_: bool;
    var phi_36775_: bool;
    var phi_14184_: f32;
    var phi_14185_: bool;
    var phi_36774_: bool;
    var phi_14186_: f32;
    var phi_14187_: bool;
    var phi_36773_: bool;
    var phi_11177_: f32;
    var phi_11179_: bool;
    var phi_36772_: bool;
    var phi_14201_: f32;
    var local_29_: f32;
    var phi_14229_: bool;
    var phi_27319_: bool;
    var phi_27379_: bool;
    var phi_27439_: bool;
    var phi_27499_: bool;
    var phi_27559_: bool;
    var phi_27619_: bool;
    var phi_27679_: bool;
    var phi_27739_: bool;
    var phi_14678_: bool;
    var phi_14726_: bool;
    var phi_14742_: bool;
    var phi_14758_: bool;
    var phi_14775_: bool;
    var phi_27799_: bool;
    var phi_14788_: f32;
    var phi_14805_: f32;
    var phi_14924_: bool;
    var phi_14930_: f32;
    var phi_14931_: f32;
    var phi_14850_: f32;
    var phi_14852_: f32;
    var phi_14853_: bool;
    var phi_14932_: f32;
    var phi_27814_: bool;
    var phi_27874_: bool;
    var phi_15136_: f32;
    var phi_15144_: f32;
    var phi_15273_: bool;
    var phi_15279_: f32;
    var phi_15280_: f32;
    var phi_15199_: f32;
    var phi_15201_: f32;
    var phi_15202_: bool;
    var phi_15222_: f32;
    var phi_15281_: f32;
    var phi_15284_: type_29_;
    var local_30_: f32;
    var local_31_: f32;
    var local_32_: f32;
    var phi_27994_: bool;
    var phi_15400_: bool;
    var phi_28054_: bool;
    var phi_15365_: bool;
    var phi_15406_: f32;
    var phi_15407_: f32;
    var phi_15408_: f32;
    var phi_15409_: bool;
    var phi_15415_: f32;
    var phi_15417_: f32;
    var phi_15418_: bool;
    var phi_15423_: f32;
    var phi_15461_: u32;
    var phi_15462_: u32;
    var phi_36776_: bool;
    var phi_16214_: bool;
    var phi_16245_: bool;
    var phi_15541_: u32;
    var phi_15551_: type_15_;
    var phi_15554_: f32;
    var phi_15552_: type_15_;
    var phi_15577_: type_15_;
    var phi_15622_: bool;
    var phi_28114_: bool;
    var phi_28174_: bool;
    var phi_28234_: bool;
    var phi_28294_: bool;
    var phi_28354_: bool;
    var phi_28414_: bool;
    var phi_28474_: bool;
    var phi_28534_: bool;
    var phi_16071_: bool;
    var phi_15555_: f32;
    var phi_36929_: bool;
    var local_33_: f32;
    var phi_16129_: bool;
    var phi_36992_: bool;
    var phi_36983_: bool;
    var phi_36982_: bool;
    var phi_36981_: bool;
    var phi_36980_: bool;
    var phi_36999_: bool;
    var phi_17951_: type_15_;
    var phi_17952_: type_15_;
    var phi_17975_: type_15_;
    var phi_36995_: bool;
    var phi_28594_: bool;
    var phi_28609_: bool;
    var phi_28624_: bool;
    var phi_28639_: bool;
    var phi_28654_: bool;
    var phi_28669_: bool;
    var phi_28684_: bool;
    var phi_28699_: bool;
    var phi_28714_: bool;
    var phi_28729_: bool;
    var phi_28744_: bool;
    var phi_28759_: bool;
    var phi_28774_: bool;
    var phi_28789_: bool;
    var phi_28804_: bool;
    var phi_28819_: bool;
    var phi_28834_: bool;
    var phi_28849_: bool;
    var phi_28864_: bool;
    var phi_28879_: bool;
    var phi_28894_: bool;
    var phi_28909_: bool;
    var phi_28924_: bool;
    var phi_28939_: bool;
    var phi_18295_: bool;
    var phi_18312_: bool;
    var phi_18329_: bool;
    var phi_18346_: bool;
    var phi_28954_: bool;
    var phi_28969_: bool;
    var phi_28984_: bool;
    var phi_28999_: bool;
    var phi_29014_: bool;
    var phi_29029_: bool;
    var phi_29044_: bool;
    var phi_29059_: bool;
    var phi_29074_: bool;
    var phi_29089_: bool;
    var phi_29104_: bool;
    var phi_29119_: bool;
    var phi_29134_: bool;
    var phi_29149_: bool;
    var phi_29164_: bool;
    var phi_29179_: bool;
    var phi_29194_: bool;
    var phi_29209_: bool;
    var phi_29224_: bool;
    var phi_29239_: bool;
    var phi_29254_: bool;
    var phi_29269_: bool;
    var phi_29284_: bool;
    var phi_29299_: bool;
    var phi_29314_: bool;
    var phi_29329_: bool;
    var phi_29344_: bool;
    var phi_29359_: bool;
    var phi_29374_: bool;
    var phi_29389_: bool;
    var phi_29404_: bool;
    var phi_29419_: bool;
    var phi_29434_: bool;
    var phi_29449_: bool;
    var phi_29464_: bool;
    var phi_29479_: bool;
    var phi_29494_: bool;
    var phi_29509_: bool;
    var phi_29524_: bool;
    var phi_29539_: bool;
    var phi_29554_: bool;
    var phi_29569_: bool;
    var phi_29584_: bool;
    var phi_29599_: bool;
    var phi_29614_: bool;
    var phi_29629_: bool;
    var phi_29644_: bool;
    var phi_29659_: bool;
    var phi_29674_: bool;
    var phi_29689_: bool;
    var phi_29704_: bool;
    var phi_29719_: bool;
    var phi_18630_: bool;
    var phi_18647_: bool;
    var phi_18664_: bool;
    var phi_18681_: bool;
    var phi_29734_: bool;
    var phi_29749_: bool;
    var phi_29764_: bool;
    var phi_29779_: bool;
    var phi_29794_: bool;
    var phi_29809_: bool;
    var phi_29824_: bool;
    var phi_29839_: bool;
    var phi_29854_: bool;
    var phi_29869_: bool;
    var phi_29884_: bool;
    var phi_29899_: bool;
    var phi_29914_: bool;
    var phi_29929_: bool;
    var phi_29944_: bool;
    var phi_29959_: bool;
    var phi_29974_: bool;
    var phi_29989_: bool;
    var phi_30004_: bool;
    var phi_30019_: bool;
    var phi_30034_: bool;
    var phi_30049_: bool;
    var phi_30064_: bool;
    var phi_30079_: bool;
    var phi_30094_: bool;
    var phi_30109_: bool;
    var phi_30124_: bool;
    var phi_30139_: bool;
    var phi_30154_: bool;
    var phi_30169_: bool;
    var phi_30184_: bool;
    var phi_30199_: bool;
    var phi_30214_: bool;
    var phi_30229_: bool;
    var phi_30244_: bool;
    var phi_30259_: bool;
    var phi_30274_: bool;
    var phi_30289_: bool;
    var phi_30304_: bool;
    var phi_30319_: bool;
    var phi_30334_: bool;
    var phi_30349_: bool;
    var phi_30364_: bool;
    var phi_30379_: bool;
    var phi_18979_: type_15_;
    var phi_18980_: type_15_;
    var phi_19003_: type_15_;
    var phi_37004_: bool;
    var phi_37309_: bool;
    var phi_19042_: type_32_;
    var phi_19043_: type_32_;
    var phi_19044_: type_32_;
    var phi_19045_: type_32_;
    var phi_19046_: type_32_;
    var phi_19047_: type_32_;
    var phi_19048_: type_32_;
    var phi_19049_: type_32_;
    var phi_37308_: bool;
    var phi_19050_: type_32_;
    var phi_19051_: type_32_;
    var phi_19052_: type_32_;
    var phi_19053_: type_32_;
    var phi_19054_: type_32_;
    var phi_19055_: type_32_;
    var phi_19056_: type_32_;
    var phi_19057_: type_32_;
    var phi_19134_: type_15_;
    var phi_19135_: type_15_;
    var phi_19158_: type_15_;
    var phi_37304_: bool;
    var phi_19071_: type_15_;
    var phi_19072_: type_15_;
    var phi_19095_: type_15_;
    var phi_37310_: bool;
    var phi_38276_: bool;
    var phi_30394_: bool;
    var phi_30409_: bool;
    var phi_30424_: bool;
    var phi_30439_: bool;
    var phi_30454_: bool;
    var phi_30469_: bool;
    var phi_30484_: bool;
    var phi_30499_: bool;
    var phi_30514_: bool;
    var phi_30529_: bool;
    var phi_30544_: bool;
    var phi_30559_: bool;
    var phi_30574_: bool;
    var phi_30589_: bool;
    var phi_30604_: bool;
    var phi_30619_: bool;
    var phi_30634_: bool;
    var phi_30649_: bool;
    var phi_30664_: bool;
    var phi_30679_: bool;
    var phi_30694_: bool;
    var phi_30709_: bool;
    var phi_30724_: bool;
    var phi_30739_: bool;
    var phi_30754_: bool;
    var phi_30769_: bool;
    var phi_30784_: bool;
    var phi_30799_: bool;
    var phi_30814_: bool;
    var phi_30829_: bool;
    var phi_30844_: bool;
    var phi_30859_: bool;
    var phi_30874_: bool;
    var phi_30889_: bool;
    var phi_30904_: bool;
    var phi_30919_: bool;
    var phi_30934_: bool;
    var phi_30949_: bool;
    var phi_30964_: bool;
    var phi_30979_: bool;
    var phi_30994_: bool;
    var phi_31009_: bool;
    var phi_31024_: bool;
    var phi_31039_: bool;
    var phi_31054_: bool;
    var phi_31069_: bool;
    var phi_31084_: bool;
    var phi_31099_: bool;
    var phi_31114_: bool;
    var phi_31129_: bool;
    var phi_31144_: bool;
    var phi_31159_: bool;
    var phi_19912_: bool;
    var phi_19914_: bool;
    var phi_19915_: bool;
    var phi_19916_: bool;
    var phi_19917_: bool;
    var phi_31174_: bool;
    var phi_31189_: bool;
    var phi_31204_: bool;
    var phi_31219_: bool;
    var phi_31234_: bool;
    var phi_31249_: bool;
    var phi_31264_: bool;
    var phi_31279_: bool;
    var phi_31294_: bool;
    var phi_31309_: bool;
    var phi_31324_: bool;
    var phi_31339_: bool;
    var phi_31354_: bool;
    var phi_31369_: bool;
    var phi_31384_: bool;
    var phi_31399_: bool;
    var phi_31414_: bool;
    var phi_31429_: bool;
    var phi_31444_: bool;
    var phi_31459_: bool;
    var phi_31474_: bool;
    var phi_31489_: bool;
    var phi_31504_: bool;
    var phi_31519_: bool;
    var phi_31534_: bool;
    var phi_31549_: bool;
    var phi_31564_: bool;
    var phi_31579_: bool;
    var phi_31594_: bool;
    var phi_31609_: bool;
    var phi_31624_: bool;
    var phi_31639_: bool;
    var phi_31654_: bool;
    var phi_31669_: bool;
    var phi_31684_: bool;
    var phi_31699_: bool;
    var phi_31714_: bool;
    var phi_31729_: bool;
    var phi_31744_: bool;
    var phi_31759_: bool;
    var phi_31774_: bool;
    var phi_31789_: bool;
    var phi_31804_: bool;
    var phi_31819_: bool;
    var phi_31834_: bool;
    var phi_31849_: bool;
    var phi_31864_: bool;
    var phi_31879_: bool;
    var phi_31894_: bool;
    var phi_31909_: bool;
    var phi_31924_: bool;
    var phi_31939_: bool;
    var phi_32194_: bool;
    var phi_32209_: bool;
    var phi_32224_: bool;
    var phi_32239_: bool;
    var phi_20327_: bool;
    var phi_20329_: bool;
    var phi_20330_: bool;
    var phi_20331_: bool;
    var phi_20332_: bool;
    var phi_32254_: bool;
    var phi_32269_: bool;
    var phi_32284_: bool;
    var phi_32299_: bool;
    var phi_20569_: type_32_;
    var phi_20605_: type_33_;
    var phi_32314_: bool;
    var phi_32329_: bool;
    var phi_32344_: bool;
    var phi_32359_: bool;
    var phi_32374_: bool;
    var phi_32389_: bool;
    var phi_32404_: bool;
    var phi_32419_: bool;
    var phi_32434_: bool;
    var phi_32449_: bool;
    var phi_32464_: bool;
    var phi_32479_: bool;
    var phi_32494_: bool;
    var phi_32509_: bool;
    var phi_32524_: bool;
    var phi_32539_: bool;
    var phi_32554_: bool;
    var phi_32569_: bool;
    var phi_32584_: bool;
    var phi_32599_: bool;
    var phi_32614_: bool;
    var phi_32629_: bool;
    var phi_32644_: bool;
    var phi_32659_: bool;
    var phi_32674_: bool;
    var phi_32689_: bool;
    var phi_32704_: bool;
    var phi_32719_: bool;
    var phi_32734_: bool;
    var phi_32749_: bool;
    var phi_32764_: bool;
    var phi_32779_: bool;
    var phi_32794_: bool;
    var phi_32809_: bool;
    var phi_32824_: bool;
    var phi_32839_: bool;
    var phi_32854_: bool;
    var phi_32869_: bool;
    var phi_32884_: bool;
    var phi_32899_: bool;
    var phi_32914_: bool;
    var phi_32929_: bool;
    var phi_32944_: bool;
    var phi_32959_: bool;
    var phi_32974_: bool;
    var phi_32989_: bool;
    var phi_33004_: bool;
    var phi_33019_: bool;
    var phi_33034_: bool;
    var phi_33049_: bool;
    var phi_33064_: bool;
    var phi_33079_: bool;
    var phi_21324_: bool;
    var phi_21326_: bool;
    var phi_21327_: bool;
    var phi_21328_: bool;
    var phi_21329_: bool;
    var phi_33094_: bool;
    var phi_33109_: bool;
    var phi_33124_: bool;
    var phi_33139_: bool;
    var phi_33154_: bool;
    var phi_33169_: bool;
    var phi_33184_: bool;
    var phi_33199_: bool;
    var phi_33214_: bool;
    var phi_33229_: bool;
    var phi_33244_: bool;
    var phi_33259_: bool;
    var phi_33274_: bool;
    var phi_33289_: bool;
    var phi_33304_: bool;
    var phi_33319_: bool;
    var phi_33334_: bool;
    var phi_33349_: bool;
    var phi_33364_: bool;
    var phi_33379_: bool;
    var phi_33394_: bool;
    var phi_33409_: bool;
    var phi_33424_: bool;
    var phi_33439_: bool;
    var phi_33454_: bool;
    var phi_33469_: bool;
    var phi_33484_: bool;
    var phi_33499_: bool;
    var phi_33514_: bool;
    var phi_33529_: bool;
    var phi_33544_: bool;
    var phi_33559_: bool;
    var phi_33574_: bool;
    var phi_33589_: bool;
    var phi_33604_: bool;
    var phi_33619_: bool;
    var phi_33634_: bool;
    var phi_33649_: bool;
    var phi_33664_: bool;
    var phi_33679_: bool;
    var phi_33694_: bool;
    var phi_33709_: bool;
    var phi_33724_: bool;
    var phi_33739_: bool;
    var phi_33754_: bool;
    var phi_33769_: bool;
    var phi_33784_: bool;
    var phi_33799_: bool;
    var phi_33814_: bool;
    var phi_33829_: bool;
    var phi_33844_: bool;
    var phi_33859_: bool;
    var phi_34114_: bool;
    var phi_34129_: bool;
    var phi_34144_: bool;
    var phi_34159_: bool;
    var phi_21739_: bool;
    var phi_21741_: bool;
    var phi_21742_: bool;
    var phi_21743_: bool;
    var phi_21744_: bool;
    var phi_34174_: bool;
    var phi_34189_: bool;
    var phi_34204_: bool;
    var phi_34219_: bool;
    var phi_21981_: type_32_;
    var phi_22017_: type_33_;
    var phi_34234_: bool;
    var phi_34249_: bool;
    var phi_34264_: bool;
    var phi_34279_: bool;
    var phi_34294_: bool;
    var phi_34309_: bool;
    var phi_34324_: bool;
    var phi_34339_: bool;
    var phi_34354_: bool;
    var phi_34369_: bool;
    var phi_34384_: bool;
    var phi_34399_: bool;
    var phi_34414_: bool;
    var phi_34429_: bool;
    var phi_34444_: bool;
    var phi_34459_: bool;
    var phi_22122_: bool;
    var phi_22139_: bool;
    var phi_22156_: bool;
    var phi_22173_: bool;
    var phi_34474_: bool;
    var phi_34489_: bool;
    var phi_34504_: bool;
    var phi_34519_: bool;
    var phi_34534_: bool;
    var phi_34549_: bool;
    var phi_34564_: bool;
    var phi_34579_: bool;
    var phi_34594_: bool;
    var phi_34609_: bool;
    var phi_34624_: bool;
    var phi_34639_: bool;
    var phi_34654_: bool;
    var phi_34669_: bool;
    var phi_34684_: bool;
    var phi_34699_: bool;
    var phi_34714_: bool;
    var phi_34729_: bool;
    var phi_34744_: bool;
    var phi_34759_: bool;
    var phi_34774_: bool;
    var phi_34789_: bool;
    var phi_34804_: bool;
    var phi_34819_: bool;
    var phi_34834_: bool;
    var phi_34849_: bool;
    var phi_34864_: bool;
    var phi_34879_: bool;
    var phi_34894_: bool;
    var phi_34909_: bool;
    var phi_34924_: bool;
    var phi_34939_: bool;
    var phi_22392_: bool;
    var phi_22409_: bool;
    var phi_22426_: bool;
    var phi_22443_: bool;
    var phi_34954_: bool;
    var phi_34969_: bool;
    var phi_34984_: bool;
    var phi_34999_: bool;
    var phi_35014_: bool;
    var phi_35029_: bool;
    var phi_35044_: bool;
    var phi_35059_: bool;
    var phi_35074_: bool;
    var phi_35089_: bool;
    var phi_35104_: bool;
    var phi_35119_: bool;
    var phi_35134_: bool;
    var phi_35149_: bool;
    var phi_35164_: bool;
    var phi_35179_: bool;
    var phi_22645_: bool;
    var phi_22661_: bool;
    var phi_22677_: bool;
    var phi_22693_: bool;
    var phi_22717_: bool;
    var phi_22733_: bool;
    var phi_22749_: bool;
    var phi_22765_: bool;
    var phi_35194_: bool;
    var phi_35209_: bool;
    var phi_35224_: bool;
    var phi_35239_: bool;
    var phi_35254_: bool;
    var phi_35269_: bool;
    var phi_35284_: bool;
    var phi_35299_: bool;
    var phi_35314_: bool;
    var phi_35329_: bool;
    var phi_35344_: bool;
    var phi_35359_: bool;
    var phi_35374_: bool;
    var phi_35389_: bool;
    var phi_35404_: bool;
    var phi_35419_: bool;
    var phi_35434_: bool;
    var phi_35449_: bool;
    var phi_35464_: bool;
    var phi_35479_: bool;
    var phi_35494_: bool;
    var phi_35509_: bool;
    var phi_35524_: bool;
    var phi_35539_: bool;
    var phi_23222_: bool;
    var phi_23239_: bool;
    var phi_23256_: bool;
    var phi_23273_: bool;
    var phi_23293_: bool;
    var phi_23310_: bool;
    var phi_23327_: bool;
    var phi_23344_: bool;
    var phi_23377_: bool;
    var phi_23393_: bool;
    var phi_23409_: bool;
    var phi_23425_: bool;
    var phi_23458_: bool;
    var phi_23474_: bool;
    var phi_23490_: bool;
    var phi_23506_: bool;
    var phi_23534_: bool;
    var phi_23550_: bool;
    var phi_23566_: bool;
    var phi_23582_: bool;
    var phi_23610_: bool;
    var phi_23626_: bool;
    var phi_23642_: bool;
    var phi_23658_: bool;
    var phi_23690_: bool;
    var phi_23706_: bool;
    var phi_23722_: bool;
    var phi_23738_: bool;
    var phi_23770_: bool;
    var phi_23786_: bool;
    var phi_23802_: bool;
    var phi_23818_: bool;
    var phi_23852_: type_15_;
    var phi_23853_: type_15_;
    var phi_23876_: type_15_;
    var phi_37314_: bool;
    var phi_38290_: bool;
    var phi_38289_: bool;
    var phi_38288_: bool;
    var phi_38287_: bool;
    var phi_38286_: bool;
    var phi_38285_: bool;
    var phi_38292_: bool;
    var phi_38291_: bool;
    var local_34_: u32;
    var local_35_: u32;
    var local_36_: f32;
    var local_37_: bool;

    switch bitcast<i32>(0u) {
        default: {
            let _e185_ = wg_id_1_;
            let _e186_ = num_wg_1_;
            let _e187_ = lid_1_;
            let _e189_ = arrayLength((&core_inputs.member));
            let _e191_ = arrayLength((&quad_params.member));
            let _e193_ = arrayLength((&tile_checkpoints.member));
            let _e195_ = arrayLength((&feature_outputs.member));
            let _e196_ = (_e187_ & 255u);
            let _e197_ = (_e196_ & 127u);
            let _e200_ = ((_e196_ >> bitcast<u32>(7i)) != 0u);
            let _e205_ = select(1u, _e186_.y, (1u < _e186_.y));
            let _e209_ = uniforms.member.member_2_;
            let _e211_ = select(1u, _e209_, (1u < _e209_));
            let _e214_ = uniforms.member.member_7_;
            let _e215_ = (_e214_ == 4294967295u);
            if _e215_ {
                phi_304_ = select(16u, _e185_.x, (_e185_.x < _e211_));
            } else {
                let _e216_ = (_e185_.x < 4u);
                if _e216_ {
                    local_12_ = array<array<u32, 4>, 4>(array<u32, 4>(0u, 4u, 8u, 16u), array<u32, 4>(1u, 5u, 9u, 16u), array<u32, 4>(2u, 6u, 16u, 16u), array<u32, 4>(3u, 7u, 16u, 16u));
                    let _e217_ = (_e214_ & 3u);
                    if (_e217_ < 4u) {
                    } else {
                        break;
                    }
                    if _e216_ {
                    } else {
                        break;
                    }
                    let _e221_ = local_12_[_e217_][_e185_.x];
                    phi_303_ = select(16u, _e221_, (_e221_ < _e211_));
                } else {
                    phi_303_ = 16u;
                }
                let _e225_ = phi_303_;
                phi_304_ = _e225_;
            }
            let _e229_ = phi_304_;
            if (_e229_ >= _e211_) {
            } else {
                let _e232_ = ((_e185_.z * _e211_) + _e229_);
                if _e215_ {
                    phi_321_ = false;
                } else {
                    phi_321_ = ((_e214_ & 1073741824u) != 0u);
                }
                let _e236_ = phi_321_;
                if _e215_ {
                    phi_329_ = false;
                } else {
                    phi_329_ = ((_e214_ & 3u) == 0u);
                }
                let _e240_ = phi_329_;
                let _e243_ = uniforms.member.member;
                if (_e243_ > 0u) {
                    phi_339_ = (((_e243_ - 1u) >> bitcast<u32>(8i)) + 1u);
                } else {
                    phi_339_ = 0u;
                }
                let _e250_ = phi_339_;
                let _e253_ = uniforms.member.member_1_;
                let _e255_ = select(1u, _e253_, (1u < _e253_));
                let _e257_ = select(_e253_, _e250_, (_e250_ < _e253_));
                let _e260_ = uniforms.member.member_4_;
                let _e262_ = select(_e260_, _e257_, (_e257_ < _e260_));
                let _e263_ = (_e205_ <= 1u);
                if _e263_ {
                    phi_437_ = _e262_;
                    phi_438_ = _e257_;
                } else {
                    if (_e257_ > _e262_) {
                        phi_370_ = (_e257_ - _e262_);
                    } else {
                        phi_370_ = 0u;
                    }
                    let _e267_ = phi_370_;
                    if (_e267_ == 0u) {
                        phi_423_ = 0u;
                    } else {
                        if _e263_ {
                            phi_422_ = _e267_;
                        } else {
                            let _e269_ = (_e267_ - 1u);
                            phi_381_ = 1u;
                            phi_384_ = _e205_;
                            loop {
                                let _e271_ = phi_381_;
                                let _e273_ = phi_384_;
                                local_34_ = _e271_;
                                local_35_ = _e273_;
                                let _e276_ = (_e273_ <= (_e269_ >> bitcast<u32>(1i)));
                                if _e276_ {
                                    phi_382_ = (_e271_ << bitcast<u32>(1i));
                                    phi_385_ = (_e273_ << bitcast<u32>(1i));
                                } else {
                                    phi_382_ = u32();
                                    phi_385_ = u32();
                                }
                                let _e282_ = phi_382_;
                                let _e284_ = phi_385_;
                                phi_381_ = _e282_;
                                phi_384_ = _e284_;
                                if !(_e276_) {
                                    break;
                                }
                            }
                            phi_396_ = 0u;
                            let _e11272_ = local_34_;
                            phi_399_ = _e11272_;
                            phi_401_ = _e269_;
                            let _e11276_ = local_35_;
                            phi_403_ = _e11276_;
                            loop {
                                let _e287_ = phi_396_;
                                let _e289_ = phi_399_;
                                let _e291_ = phi_401_;
                                let _e293_ = phi_403_;
                                local_23_ = _e287_;
                                let _e294_ = (_e289_ > 0u);
                                if _e294_ {
                                    if (_e291_ >= _e293_) {
                                        phi_417_ = (_e287_ | _e289_);
                                        phi_418_ = (_e291_ - _e293_);
                                    } else {
                                        phi_417_ = _e287_;
                                        phi_418_ = _e291_;
                                    }
                                    let _e299_ = phi_417_;
                                    let _e301_ = phi_418_;
                                    phi_397_ = _e299_;
                                    phi_400_ = (_e289_ >> bitcast<u32>(1i));
                                    phi_402_ = _e301_;
                                    phi_404_ = (_e293_ >> bitcast<u32>(1i));
                                } else {
                                    phi_397_ = u32();
                                    phi_400_ = u32();
                                    phi_402_ = u32();
                                    phi_404_ = u32();
                                }
                                let _e307_ = phi_397_;
                                let _e309_ = phi_400_;
                                let _e311_ = phi_402_;
                                let _e313_ = phi_404_;
                                phi_396_ = _e307_;
                                phi_399_ = _e309_;
                                phi_401_ = _e311_;
                                phi_403_ = _e313_;
                                if !(_e294_) {
                                    break;
                                }
                            }
                            let _e316_ = local_23_;
                            phi_422_ = (_e316_ + 1u);
                        }
                        let _e319_ = phi_422_;
                        phi_423_ = _e319_;
                    }
                    let _e321_ = phi_423_;
                    let _e323_ = (_e262_ + (_e185_.y * _e321_));
                    let _e325_ = select(_e323_, _e257_, (_e257_ < _e323_));
                    let _e326_ = (_e325_ + _e321_);
                    phi_437_ = _e325_;
                    phi_438_ = select(_e326_, _e257_, (_e257_ < _e326_));
                }
                let _e330_ = phi_437_;
                let _e332_ = phi_438_;
                let _e333_ = (_e232_ * 28u);
                let _e334_ = (_e333_ < _e191_);
                if _e334_ {
                    if _e334_ {
                    } else {
                        break;
                    }
                    let _e337_ = quad_params.member[_e333_];
                    phi_452_ = _e337_;
                } else {
                    phi_452_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e339_ = phi_452_;
                let _e340_ = (_e333_ + 1u);
                let _e341_ = (_e340_ < _e191_);
                if _e341_ {
                    if _e341_ {
                    } else {
                        break;
                    }
                    let _e344_ = quad_params.member[_e340_];
                    phi_465_ = _e344_;
                } else {
                    phi_465_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e346_ = phi_465_;
                let _e347_ = (_e333_ + 2u);
                let _e348_ = (_e347_ < _e191_);
                if _e348_ {
                    if _e348_ {
                    } else {
                        break;
                    }
                    let _e351_ = quad_params.member[_e347_];
                    phi_478_ = _e351_;
                } else {
                    phi_478_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e353_ = phi_478_;
                let _e354_ = (_e333_ + 3u);
                let _e355_ = (_e354_ < _e191_);
                if _e355_ {
                    if _e355_ {
                    } else {
                        break;
                    }
                    let _e358_ = quad_params.member[_e354_];
                    phi_491_ = _e358_;
                } else {
                    phi_491_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e360_ = phi_491_;
                let _e361_ = (_e333_ + 4u);
                let _e362_ = (_e361_ < _e191_);
                if _e362_ {
                    if _e362_ {
                    } else {
                        break;
                    }
                    let _e365_ = quad_params.member[_e361_];
                    phi_504_ = _e365_;
                } else {
                    phi_504_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e367_ = phi_504_;
                let _e368_ = (_e333_ + 5u);
                let _e369_ = (_e368_ < _e191_);
                if _e369_ {
                    if _e369_ {
                    } else {
                        break;
                    }
                    let _e372_ = quad_params.member[_e368_];
                    phi_517_ = _e372_;
                } else {
                    phi_517_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e374_ = phi_517_;
                let _e375_ = (_e333_ + 6u);
                let _e376_ = (_e375_ < _e191_);
                if _e376_ {
                    if _e376_ {
                    } else {
                        break;
                    }
                    let _e379_ = quad_params.member[_e375_];
                    phi_530_ = _e379_;
                } else {
                    phi_530_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e381_ = phi_530_;
                let _e382_ = (_e333_ + 7u);
                let _e383_ = (_e382_ < _e191_);
                if _e383_ {
                    if _e383_ {
                    } else {
                        break;
                    }
                    let _e386_ = quad_params.member[_e382_];
                    phi_543_ = _e386_;
                } else {
                    phi_543_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e388_ = phi_543_;
                let _e389_ = (_e333_ + 8u);
                let _e390_ = (_e389_ < _e191_);
                if _e390_ {
                    if _e390_ {
                    } else {
                        break;
                    }
                    let _e393_ = quad_params.member[_e389_];
                    phi_556_ = _e393_;
                } else {
                    phi_556_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e395_ = phi_556_;
                let _e396_ = (_e333_ + 9u);
                let _e397_ = (_e396_ < _e191_);
                if _e397_ {
                    if _e397_ {
                    } else {
                        break;
                    }
                    let _e400_ = quad_params.member[_e396_];
                    phi_569_ = _e400_;
                } else {
                    phi_569_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e402_ = phi_569_;
                let _e403_ = (_e333_ + 10u);
                let _e404_ = (_e403_ < _e191_);
                if _e404_ {
                    if _e404_ {
                    } else {
                        break;
                    }
                    let _e407_ = quad_params.member[_e403_];
                    phi_582_ = _e407_;
                } else {
                    phi_582_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e409_ = phi_582_;
                let _e410_ = (_e333_ + 11u);
                let _e411_ = (_e410_ < _e191_);
                if _e411_ {
                    if _e411_ {
                    } else {
                        break;
                    }
                    let _e414_ = quad_params.member[_e410_];
                    phi_595_ = _e414_;
                } else {
                    phi_595_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e416_ = phi_595_;
                let _e417_ = (_e333_ + 12u);
                let _e418_ = (_e417_ < _e191_);
                if _e418_ {
                    if _e418_ {
                    } else {
                        break;
                    }
                    let _e421_ = quad_params.member[_e417_];
                    phi_608_ = _e421_;
                } else {
                    phi_608_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e423_ = phi_608_;
                let _e424_ = (_e333_ + 13u);
                let _e425_ = (_e424_ < _e191_);
                if _e425_ {
                    if _e425_ {
                    } else {
                        break;
                    }
                    let _e428_ = quad_params.member[_e424_];
                    phi_621_ = _e428_;
                } else {
                    phi_621_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e430_ = phi_621_;
                let _e431_ = (_e333_ + 14u);
                let _e432_ = (_e431_ < _e191_);
                if _e432_ {
                    if _e432_ {
                    } else {
                        break;
                    }
                    let _e435_ = quad_params.member[_e431_];
                    phi_634_ = _e435_;
                } else {
                    phi_634_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e437_ = phi_634_;
                let _e438_ = (_e333_ + 15u);
                let _e439_ = (_e438_ < _e191_);
                if _e439_ {
                    if _e439_ {
                    } else {
                        break;
                    }
                    let _e442_ = quad_params.member[_e438_];
                    phi_647_ = _e442_;
                } else {
                    phi_647_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e444_ = phi_647_;
                let _e445_ = (_e333_ + 16u);
                let _e446_ = (_e445_ < _e191_);
                if _e446_ {
                    if _e446_ {
                    } else {
                        break;
                    }
                    let _e449_ = quad_params.member[_e445_];
                    phi_660_ = _e449_;
                } else {
                    phi_660_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e451_ = phi_660_;
                let _e452_ = (_e333_ + 17u);
                let _e453_ = (_e452_ < _e191_);
                if _e453_ {
                    if _e453_ {
                    } else {
                        break;
                    }
                    let _e456_ = quad_params.member[_e452_];
                    phi_673_ = _e456_;
                } else {
                    phi_673_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e458_ = phi_673_;
                let _e459_ = (_e333_ + 18u);
                let _e460_ = (_e459_ < _e191_);
                if _e460_ {
                    if _e460_ {
                    } else {
                        break;
                    }
                    let _e463_ = quad_params.member[_e459_];
                    phi_686_ = _e463_;
                } else {
                    phi_686_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e465_ = phi_686_;
                let _e466_ = (_e333_ + 19u);
                let _e467_ = (_e466_ < _e191_);
                if _e467_ {
                    if _e467_ {
                    } else {
                        break;
                    }
                    let _e470_ = quad_params.member[_e466_];
                    phi_699_ = _e470_;
                } else {
                    phi_699_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e472_ = phi_699_;
                let _e473_ = (_e333_ + 20u);
                let _e474_ = (_e473_ < _e191_);
                if _e474_ {
                    if _e474_ {
                    } else {
                        break;
                    }
                    let _e477_ = quad_params.member[_e473_];
                    phi_712_ = _e477_;
                } else {
                    phi_712_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e479_ = phi_712_;
                let _e480_ = (_e333_ + 21u);
                let _e481_ = (_e480_ < _e191_);
                if _e481_ {
                    if _e481_ {
                    } else {
                        break;
                    }
                    let _e484_ = quad_params.member[_e480_];
                    phi_725_ = _e484_;
                } else {
                    phi_725_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e486_ = phi_725_;
                let _e487_ = (_e333_ + 22u);
                let _e488_ = (_e487_ < _e191_);
                if _e488_ {
                    if _e488_ {
                    } else {
                        break;
                    }
                    let _e491_ = quad_params.member[_e487_];
                    phi_738_ = _e491_;
                } else {
                    phi_738_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e493_ = phi_738_;
                let _e494_ = (_e333_ + 23u);
                let _e495_ = (_e494_ < _e191_);
                if _e495_ {
                    if _e495_ {
                    } else {
                        break;
                    }
                    let _e498_ = quad_params.member[_e494_];
                    phi_751_ = _e498_;
                } else {
                    phi_751_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e500_ = phi_751_;
                let _e501_ = (_e333_ + 24u);
                let _e502_ = (_e501_ < _e191_);
                if _e502_ {
                    if _e502_ {
                    } else {
                        break;
                    }
                    let _e505_ = quad_params.member[_e501_];
                    phi_764_ = _e505_;
                } else {
                    phi_764_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e507_ = phi_764_;
                let _e508_ = (_e333_ + 25u);
                let _e509_ = (_e508_ < _e191_);
                if _e509_ {
                    if _e509_ {
                    } else {
                        break;
                    }
                    let _e512_ = quad_params.member[_e508_];
                    phi_777_ = _e512_;
                } else {
                    phi_777_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e514_ = phi_777_;
                let _e515_ = (_e333_ + 26u);
                let _e516_ = (_e515_ < _e191_);
                if _e516_ {
                    if _e516_ {
                    } else {
                        break;
                    }
                    let _e519_ = quad_params.member[_e515_];
                    phi_790_ = _e519_;
                } else {
                    phi_790_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e521_ = phi_790_;
                let _e522_ = (_e333_ + 27u);
                let _e523_ = (_e522_ < _e191_);
                if _e523_ {
                    if _e523_ {
                    } else {
                        break;
                    }
                    let _e526_ = quad_params.member[_e522_];
                    phi_803_ = _e526_;
                } else {
                    phi_803_ = vec4<f32>(0f, 0f, 0f, 0f);
                }
                let _e528_ = phi_803_;
                let _e530_ = bitcast<u32>(_e430_.x);
                let _e531_ = (_e530_ & 2139095040u);
                if (_e531_ != 2139095040u) {
                    if (_e430_.x <= 0f) {
                        phi_837_ = 0f;
                    } else {
                        if (_e430_.x >= 1f) {
                            phi_836_ = 1f;
                        } else {
                            if (_e531_ == 2139095040u) {
                                phi_830_ = true;
                            } else {
                                phi_830_ = (_e531_ == 0u);
                            }
                            let _e540_ = phi_830_;
                            phi_836_ = bitcast<f32>(select(_e530_, 0u, _e540_));
                        }
                        let _e544_ = phi_836_;
                        phi_837_ = _e544_;
                    }
                    let _e546_ = phi_837_;
                    phi_844_ = _e546_;
                } else {
                    phi_844_ = select(0f, 1f, (_e530_ == 2139095040u));
                }
                let _e548_ = phi_844_;
                let _e550_ = bitcast<u32>(_e430_.y);
                let _e551_ = (_e550_ & 2139095040u);
                if (_e551_ != 2139095040u) {
                    if (_e430_.y <= 0f) {
                        phi_877_ = 0f;
                    } else {
                        if (_e430_.y >= 1f) {
                            phi_876_ = 1f;
                        } else {
                            if (_e551_ == 2139095040u) {
                                phi_870_ = true;
                            } else {
                                phi_870_ = (_e551_ == 0u);
                            }
                            let _e560_ = phi_870_;
                            phi_876_ = bitcast<f32>(select(_e550_, 0u, _e560_));
                        }
                        let _e564_ = phi_876_;
                        phi_877_ = _e564_;
                    }
                    let _e566_ = phi_877_;
                    phi_884_ = _e566_;
                } else {
                    phi_884_ = select(0f, 1f, (_e550_ == 2139095040u));
                }
                let _e568_ = phi_884_;
                let _e570_ = bitcast<u32>(_e430_.z);
                let _e571_ = (_e570_ & 2139095040u);
                if (_e571_ != 2139095040u) {
                    if (_e430_.z <= 0f) {
                        phi_917_ = 0f;
                    } else {
                        if (_e430_.z >= 1f) {
                            phi_916_ = 1f;
                        } else {
                            if (_e571_ == 2139095040u) {
                                phi_910_ = true;
                            } else {
                                phi_910_ = (_e571_ == 0u);
                            }
                            let _e580_ = phi_910_;
                            phi_916_ = bitcast<f32>(select(_e570_, 0u, _e580_));
                        }
                        let _e584_ = phi_916_;
                        phi_917_ = _e584_;
                    }
                    let _e586_ = phi_917_;
                    phi_924_ = _e586_;
                } else {
                    phi_924_ = select(0f, 1f, (_e570_ == 2139095040u));
                }
                let _e588_ = phi_924_;
                let _e590_ = bitcast<u32>(_e430_.w);
                let _e591_ = (_e590_ & 2139095040u);
                if (_e591_ != 2139095040u) {
                    if (_e430_.w <= 0f) {
                        phi_957_ = 0f;
                    } else {
                        if (_e430_.w >= 1f) {
                            phi_956_ = 1f;
                        } else {
                            if (_e591_ == 2139095040u) {
                                phi_950_ = true;
                            } else {
                                phi_950_ = (_e591_ == 0u);
                            }
                            let _e600_ = phi_950_;
                            phi_956_ = bitcast<f32>(select(_e590_, 0u, _e600_));
                        }
                        let _e604_ = phi_956_;
                        phi_957_ = _e604_;
                    }
                    let _e606_ = phi_957_;
                    phi_964_ = _e606_;
                } else {
                    phi_964_ = select(0f, 1f, (_e590_ == 2139095040u));
                }
                let _e608_ = phi_964_;
                let _e630_ = bitcast<u32>(_e493_.x);
                let _e631_ = (_e630_ & 2139095040u);
                if (_e631_ != 2139095040u) {
                    if (_e493_.x <= -32f) {
                        phi_1050_ = -32f;
                    } else {
                        if (_e493_.x >= 32f) {
                            phi_1049_ = 32f;
                        } else {
                            if (_e631_ == 2139095040u) {
                                phi_1043_ = true;
                            } else {
                                phi_1043_ = (_e631_ == 0u);
                            }
                            let _e638_ = phi_1043_;
                            phi_1049_ = bitcast<f32>(select(_e630_, 0u, _e638_));
                        }
                        let _e642_ = phi_1049_;
                        phi_1050_ = _e642_;
                    }
                    let _e644_ = phi_1050_;
                    phi_1051_ = _e644_;
                } else {
                    phi_1051_ = 0f;
                }
                let _e646_ = phi_1051_;
                let _e648_ = bitcast<u32>(_e493_.y);
                let _e649_ = (_e648_ & 2139095040u);
                if (_e649_ != 2139095040u) {
                    if (_e493_.y <= -32f) {
                        phi_1116_ = -32f;
                    } else {
                        if (_e493_.y >= 32f) {
                            phi_1115_ = 32f;
                        } else {
                            if (_e649_ == 2139095040u) {
                                phi_1109_ = true;
                            } else {
                                phi_1109_ = (_e649_ == 0u);
                            }
                            let _e656_ = phi_1109_;
                            phi_1115_ = bitcast<f32>(select(_e648_, 0u, _e656_));
                        }
                        let _e660_ = phi_1115_;
                        phi_1116_ = _e660_;
                    }
                    let _e662_ = phi_1116_;
                    phi_1117_ = _e662_;
                } else {
                    phi_1117_ = 0f;
                }
                let _e664_ = phi_1117_;
                let _e666_ = bitcast<u32>(_e493_.z);
                let _e667_ = (_e666_ & 2139095040u);
                if (_e667_ != 2139095040u) {
                    if (_e493_.z <= -32f) {
                        phi_1182_ = -32f;
                    } else {
                        if (_e493_.z >= 32f) {
                            phi_1181_ = 32f;
                        } else {
                            if (_e667_ == 2139095040u) {
                                phi_1175_ = true;
                            } else {
                                phi_1175_ = (_e667_ == 0u);
                            }
                            let _e674_ = phi_1175_;
                            phi_1181_ = bitcast<f32>(select(_e666_, 0u, _e674_));
                        }
                        let _e678_ = phi_1181_;
                        phi_1182_ = _e678_;
                    }
                    let _e680_ = phi_1182_;
                    phi_1183_ = _e680_;
                } else {
                    phi_1183_ = 0f;
                }
                let _e682_ = phi_1183_;
                let _e684_ = bitcast<u32>(_e493_.w);
                let _e685_ = (_e684_ & 2139095040u);
                if (_e685_ != 2139095040u) {
                    if (_e493_.w <= -32f) {
                        phi_1248_ = -32f;
                    } else {
                        if (_e493_.w >= 32f) {
                            phi_1247_ = 32f;
                        } else {
                            if (_e685_ == 2139095040u) {
                                phi_1241_ = true;
                            } else {
                                phi_1241_ = (_e685_ == 0u);
                            }
                            let _e692_ = phi_1241_;
                            phi_1247_ = bitcast<f32>(select(_e684_, 0u, _e692_));
                        }
                        let _e696_ = phi_1247_;
                        phi_1248_ = _e696_;
                    }
                    let _e698_ = phi_1248_;
                    phi_1249_ = _e698_;
                } else {
                    phi_1249_ = 0f;
                }
                let _e700_ = phi_1249_;
                let _e702_ = bitcast<u32>(_e500_.x);
                let _e703_ = (_e702_ & 2139095040u);
                if (_e703_ != 2139095040u) {
                    if (_e500_.x <= -16f) {
                        phi_1315_ = -16f;
                    } else {
                        if (_e500_.x >= 16f) {
                            phi_1314_ = 16f;
                        } else {
                            if (_e703_ == 2139095040u) {
                                phi_1308_ = true;
                            } else {
                                phi_1308_ = (_e703_ == 0u);
                            }
                            let _e710_ = phi_1308_;
                            phi_1314_ = bitcast<f32>(select(_e702_, 0u, _e710_));
                        }
                        let _e714_ = phi_1314_;
                        phi_1315_ = _e714_;
                    }
                    let _e716_ = phi_1315_;
                    phi_1316_ = _e716_;
                } else {
                    phi_1316_ = 0f;
                }
                let _e718_ = phi_1316_;
                let _e720_ = bitcast<u32>(_e500_.y);
                let _e721_ = (_e720_ & 2139095040u);
                if (_e721_ != 2139095040u) {
                    if (_e500_.y <= -16f) {
                        phi_1381_ = -16f;
                    } else {
                        if (_e500_.y >= 16f) {
                            phi_1380_ = 16f;
                        } else {
                            if (_e721_ == 2139095040u) {
                                phi_1374_ = true;
                            } else {
                                phi_1374_ = (_e721_ == 0u);
                            }
                            let _e728_ = phi_1374_;
                            phi_1380_ = bitcast<f32>(select(_e720_, 0u, _e728_));
                        }
                        let _e732_ = phi_1380_;
                        phi_1381_ = _e732_;
                    }
                    let _e734_ = phi_1381_;
                    phi_1382_ = _e734_;
                } else {
                    phi_1382_ = 0f;
                }
                let _e736_ = phi_1382_;
                let _e738_ = bitcast<u32>(_e500_.z);
                let _e739_ = (_e738_ & 2139095040u);
                if (_e739_ != 2139095040u) {
                    if (_e500_.z <= -16f) {
                        phi_1447_ = -16f;
                    } else {
                        if (_e500_.z >= 16f) {
                            phi_1446_ = 16f;
                        } else {
                            if (_e739_ == 2139095040u) {
                                phi_1440_ = true;
                            } else {
                                phi_1440_ = (_e739_ == 0u);
                            }
                            let _e746_ = phi_1440_;
                            phi_1446_ = bitcast<f32>(select(_e738_, 0u, _e746_));
                        }
                        let _e750_ = phi_1446_;
                        phi_1447_ = _e750_;
                    }
                    let _e752_ = phi_1447_;
                    phi_1448_ = _e752_;
                } else {
                    phi_1448_ = 0f;
                }
                let _e754_ = phi_1448_;
                let _e756_ = bitcast<u32>(_e500_.w);
                let _e757_ = (_e756_ & 2139095040u);
                if (_e757_ != 2139095040u) {
                    if (_e500_.w <= -16f) {
                        phi_1513_ = -16f;
                    } else {
                        if (_e500_.w >= 16f) {
                            phi_1512_ = 16f;
                        } else {
                            if (_e757_ == 2139095040u) {
                                phi_1506_ = true;
                            } else {
                                phi_1506_ = (_e757_ == 0u);
                            }
                            let _e764_ = phi_1506_;
                            phi_1512_ = bitcast<f32>(select(_e756_, 0u, _e764_));
                        }
                        let _e768_ = phi_1512_;
                        phi_1513_ = _e768_;
                    }
                    let _e770_ = phi_1513_;
                    phi_1514_ = _e770_;
                } else {
                    phi_1514_ = 0f;
                }
                let _e772_ = phi_1514_;
                let _e774_ = bitcast<u32>(_e507_.x);
                let _e775_ = (_e774_ & 2139095040u);
                if (_e775_ != 2139095040u) {
                    if (_e507_.x <= 0f) {
                        phi_1579_ = 0f;
                    } else {
                        if (_e507_.x >= 1f) {
                            phi_1578_ = 1f;
                        } else {
                            if (_e775_ == 2139095040u) {
                                phi_1572_ = true;
                            } else {
                                phi_1572_ = (_e775_ == 0u);
                            }
                            let _e782_ = phi_1572_;
                            phi_1578_ = bitcast<f32>(select(_e774_, 0u, _e782_));
                        }
                        let _e786_ = phi_1578_;
                        phi_1579_ = _e786_;
                    }
                    let _e788_ = phi_1579_;
                    phi_1580_ = _e788_;
                } else {
                    phi_1580_ = 0f;
                }
                let _e790_ = phi_1580_;
                let _e792_ = bitcast<u32>(_e507_.y);
                let _e793_ = (_e792_ & 2139095040u);
                if (_e793_ != 2139095040u) {
                    if (_e507_.y <= 0f) {
                        phi_1645_ = 0f;
                    } else {
                        if (_e507_.y >= 1f) {
                            phi_1644_ = 1f;
                        } else {
                            if (_e793_ == 2139095040u) {
                                phi_1638_ = true;
                            } else {
                                phi_1638_ = (_e793_ == 0u);
                            }
                            let _e800_ = phi_1638_;
                            phi_1644_ = bitcast<f32>(select(_e792_, 0u, _e800_));
                        }
                        let _e804_ = phi_1644_;
                        phi_1645_ = _e804_;
                    }
                    let _e806_ = phi_1645_;
                    phi_1646_ = _e806_;
                } else {
                    phi_1646_ = 0f;
                }
                let _e808_ = phi_1646_;
                let _e810_ = bitcast<u32>(_e507_.z);
                let _e811_ = (_e810_ & 2139095040u);
                if (_e811_ != 2139095040u) {
                    if (_e507_.z <= 0f) {
                        phi_1711_ = 0f;
                    } else {
                        if (_e507_.z >= 1f) {
                            phi_1710_ = 1f;
                        } else {
                            if (_e811_ == 2139095040u) {
                                phi_1704_ = true;
                            } else {
                                phi_1704_ = (_e811_ == 0u);
                            }
                            let _e818_ = phi_1704_;
                            phi_1710_ = bitcast<f32>(select(_e810_, 0u, _e818_));
                        }
                        let _e822_ = phi_1710_;
                        phi_1711_ = _e822_;
                    }
                    let _e824_ = phi_1711_;
                    phi_1712_ = _e824_;
                } else {
                    phi_1712_ = 0f;
                }
                let _e826_ = phi_1712_;
                let _e828_ = bitcast<u32>(_e507_.w);
                let _e829_ = (_e828_ & 2139095040u);
                if (_e829_ != 2139095040u) {
                    if (_e507_.w <= 0f) {
                        phi_1777_ = 0f;
                    } else {
                        if (_e507_.w >= 1f) {
                            phi_1776_ = 1f;
                        } else {
                            if (_e829_ == 2139095040u) {
                                phi_1770_ = true;
                            } else {
                                phi_1770_ = (_e829_ == 0u);
                            }
                            let _e836_ = phi_1770_;
                            phi_1776_ = bitcast<f32>(select(_e828_, 0u, _e836_));
                        }
                        let _e840_ = phi_1776_;
                        phi_1777_ = _e840_;
                    }
                    let _e842_ = phi_1777_;
                    phi_1778_ = _e842_;
                } else {
                    phi_1778_ = 0f;
                }
                let _e844_ = phi_1778_;
                let _e846_ = bitcast<u32>(_e514_.x);
                let _e847_ = (_e846_ & 2139095040u);
                if (_e847_ != 2139095040u) {
                    if (_e514_.x <= 0f) {
                        phi_1843_ = 0f;
                    } else {
                        if (_e514_.x >= 1f) {
                            phi_1842_ = 1f;
                        } else {
                            if (_e847_ == 2139095040u) {
                                phi_1836_ = true;
                            } else {
                                phi_1836_ = (_e847_ == 0u);
                            }
                            let _e854_ = phi_1836_;
                            phi_1842_ = bitcast<f32>(select(_e846_, 0u, _e854_));
                        }
                        let _e858_ = phi_1842_;
                        phi_1843_ = _e858_;
                    }
                    let _e860_ = phi_1843_;
                    phi_1844_ = _e860_;
                } else {
                    phi_1844_ = 0f;
                }
                let _e862_ = phi_1844_;
                let _e864_ = bitcast<u32>(_e514_.y);
                let _e865_ = (_e864_ & 2139095040u);
                if (_e865_ != 2139095040u) {
                    if (_e514_.y <= 0f) {
                        phi_1909_ = 0f;
                    } else {
                        if (_e514_.y >= 1f) {
                            phi_1908_ = 1f;
                        } else {
                            if (_e865_ == 2139095040u) {
                                phi_1902_ = true;
                            } else {
                                phi_1902_ = (_e865_ == 0u);
                            }
                            let _e872_ = phi_1902_;
                            phi_1908_ = bitcast<f32>(select(_e864_, 0u, _e872_));
                        }
                        let _e876_ = phi_1908_;
                        phi_1909_ = _e876_;
                    }
                    let _e878_ = phi_1909_;
                    phi_1910_ = _e878_;
                } else {
                    phi_1910_ = 0f;
                }
                let _e880_ = phi_1910_;
                let _e882_ = bitcast<u32>(_e514_.z);
                let _e883_ = (_e882_ & 2139095040u);
                if (_e883_ != 2139095040u) {
                    if (_e514_.z <= 0f) {
                        phi_1975_ = 0f;
                    } else {
                        if (_e514_.z >= 1f) {
                            phi_1974_ = 1f;
                        } else {
                            if (_e883_ == 2139095040u) {
                                phi_1968_ = true;
                            } else {
                                phi_1968_ = (_e883_ == 0u);
                            }
                            let _e890_ = phi_1968_;
                            phi_1974_ = bitcast<f32>(select(_e882_, 0u, _e890_));
                        }
                        let _e894_ = phi_1974_;
                        phi_1975_ = _e894_;
                    }
                    let _e896_ = phi_1975_;
                    phi_1976_ = _e896_;
                } else {
                    phi_1976_ = 0f;
                }
                let _e898_ = phi_1976_;
                let _e900_ = bitcast<u32>(_e514_.w);
                let _e901_ = (_e900_ & 2139095040u);
                if (_e901_ != 2139095040u) {
                    if (_e514_.w <= 0f) {
                        phi_2041_ = 0f;
                    } else {
                        if (_e514_.w >= 1f) {
                            phi_2040_ = 1f;
                        } else {
                            if (_e901_ == 2139095040u) {
                                phi_2034_ = true;
                            } else {
                                phi_2034_ = (_e901_ == 0u);
                            }
                            let _e908_ = phi_2034_;
                            phi_2040_ = bitcast<f32>(select(_e900_, 0u, _e908_));
                        }
                        let _e912_ = phi_2040_;
                        phi_2041_ = _e912_;
                    }
                    let _e914_ = phi_2041_;
                    phi_2042_ = _e914_;
                } else {
                    phi_2042_ = 0f;
                }
                let _e916_ = phi_2042_;
                let _e918_ = bitcast<u32>(_e521_.x);
                let _e919_ = (_e918_ & 2139095040u);
                if (_e919_ != 2139095040u) {
                    if (_e521_.x <= -4f) {
                        phi_2108_ = -4f;
                    } else {
                        if (_e521_.x >= 4f) {
                            phi_2107_ = 4f;
                        } else {
                            if (_e919_ == 2139095040u) {
                                phi_2101_ = true;
                            } else {
                                phi_2101_ = (_e919_ == 0u);
                            }
                            let _e926_ = phi_2101_;
                            phi_2107_ = bitcast<f32>(select(_e918_, 0u, _e926_));
                        }
                        let _e930_ = phi_2107_;
                        phi_2108_ = _e930_;
                    }
                    let _e932_ = phi_2108_;
                    phi_2109_ = _e932_;
                } else {
                    phi_2109_ = 0f;
                }
                let _e934_ = phi_2109_;
                let _e936_ = bitcast<u32>(_e521_.y);
                let _e937_ = (_e936_ & 2139095040u);
                if (_e937_ != 2139095040u) {
                    if (_e521_.y <= -4f) {
                        phi_2174_ = -4f;
                    } else {
                        if (_e521_.y >= 4f) {
                            phi_2173_ = 4f;
                        } else {
                            if (_e937_ == 2139095040u) {
                                phi_2167_ = true;
                            } else {
                                phi_2167_ = (_e937_ == 0u);
                            }
                            let _e944_ = phi_2167_;
                            phi_2173_ = bitcast<f32>(select(_e936_, 0u, _e944_));
                        }
                        let _e948_ = phi_2173_;
                        phi_2174_ = _e948_;
                    }
                    let _e950_ = phi_2174_;
                    phi_2175_ = _e950_;
                } else {
                    phi_2175_ = 0f;
                }
                let _e952_ = phi_2175_;
                let _e954_ = bitcast<u32>(_e521_.z);
                let _e955_ = (_e954_ & 2139095040u);
                if (_e955_ != 2139095040u) {
                    if (_e521_.z <= -4f) {
                        phi_2240_ = -4f;
                    } else {
                        if (_e521_.z >= 4f) {
                            phi_2239_ = 4f;
                        } else {
                            if (_e955_ == 2139095040u) {
                                phi_2233_ = true;
                            } else {
                                phi_2233_ = (_e955_ == 0u);
                            }
                            let _e962_ = phi_2233_;
                            phi_2239_ = bitcast<f32>(select(_e954_, 0u, _e962_));
                        }
                        let _e966_ = phi_2239_;
                        phi_2240_ = _e966_;
                    }
                    let _e968_ = phi_2240_;
                    phi_2241_ = _e968_;
                } else {
                    phi_2241_ = 0f;
                }
                let _e970_ = phi_2241_;
                let _e972_ = bitcast<u32>(_e521_.w);
                let _e973_ = (_e972_ & 2139095040u);
                if (_e973_ != 2139095040u) {
                    if (_e521_.w <= -4f) {
                        phi_2306_ = -4f;
                    } else {
                        if (_e521_.w >= 4f) {
                            phi_2305_ = 4f;
                        } else {
                            if (_e973_ == 2139095040u) {
                                phi_2299_ = true;
                            } else {
                                phi_2299_ = (_e973_ == 0u);
                            }
                            let _e980_ = phi_2299_;
                            phi_2305_ = bitcast<f32>(select(_e972_, 0u, _e980_));
                        }
                        let _e984_ = phi_2305_;
                        phi_2306_ = _e984_;
                    }
                    let _e986_ = phi_2306_;
                    phi_2307_ = _e986_;
                } else {
                    phi_2307_ = 0f;
                }
                let _e988_ = phi_2307_;
                let _e990_ = bitcast<u32>(_e528_.x);
                let _e991_ = (_e990_ & 2139095040u);
                if (_e991_ != 2139095040u) {
                    if (_e528_.x <= -4f) {
                        phi_2373_ = -4f;
                    } else {
                        if (_e528_.x >= 4f) {
                            phi_2372_ = 4f;
                        } else {
                            if (_e991_ == 2139095040u) {
                                phi_2366_ = true;
                            } else {
                                phi_2366_ = (_e991_ == 0u);
                            }
                            let _e998_ = phi_2366_;
                            phi_2372_ = bitcast<f32>(select(_e990_, 0u, _e998_));
                        }
                        let _e1002_ = phi_2372_;
                        phi_2373_ = _e1002_;
                    }
                    let _e1004_ = phi_2373_;
                    phi_2374_ = _e1004_;
                } else {
                    phi_2374_ = 0f;
                }
                let _e1006_ = phi_2374_;
                let _e1008_ = bitcast<u32>(_e528_.y);
                let _e1009_ = (_e1008_ & 2139095040u);
                if (_e1009_ != 2139095040u) {
                    if (_e528_.y <= -4f) {
                        phi_2439_ = -4f;
                    } else {
                        if (_e528_.y >= 4f) {
                            phi_2438_ = 4f;
                        } else {
                            if (_e1009_ == 2139095040u) {
                                phi_2432_ = true;
                            } else {
                                phi_2432_ = (_e1009_ == 0u);
                            }
                            let _e1016_ = phi_2432_;
                            phi_2438_ = bitcast<f32>(select(_e1008_, 0u, _e1016_));
                        }
                        let _e1020_ = phi_2438_;
                        phi_2439_ = _e1020_;
                    }
                    let _e1022_ = phi_2439_;
                    phi_2440_ = _e1022_;
                } else {
                    phi_2440_ = 0f;
                }
                let _e1024_ = phi_2440_;
                let _e1026_ = bitcast<u32>(_e528_.z);
                let _e1027_ = (_e1026_ & 2139095040u);
                if (_e1027_ != 2139095040u) {
                    if (_e528_.z <= -4f) {
                        phi_2505_ = -4f;
                    } else {
                        if (_e528_.z >= 4f) {
                            phi_2504_ = 4f;
                        } else {
                            if (_e1027_ == 2139095040u) {
                                phi_2498_ = true;
                            } else {
                                phi_2498_ = (_e1027_ == 0u);
                            }
                            let _e1034_ = phi_2498_;
                            phi_2504_ = bitcast<f32>(select(_e1026_, 0u, _e1034_));
                        }
                        let _e1038_ = phi_2504_;
                        phi_2505_ = _e1038_;
                    }
                    let _e1040_ = phi_2505_;
                    phi_2506_ = _e1040_;
                } else {
                    phi_2506_ = 0f;
                }
                let _e1042_ = phi_2506_;
                let _e1044_ = bitcast<u32>(_e528_.w);
                let _e1045_ = (_e1044_ & 2139095040u);
                if (_e1045_ != 2139095040u) {
                    if (_e528_.w <= -4f) {
                        phi_2571_ = -4f;
                    } else {
                        if (_e528_.w >= 4f) {
                            phi_2570_ = 4f;
                        } else {
                            if (_e1045_ == 2139095040u) {
                                phi_2564_ = true;
                            } else {
                                phi_2564_ = (_e1045_ == 0u);
                            }
                            let _e1052_ = phi_2564_;
                            phi_2570_ = bitcast<f32>(select(_e1044_, 0u, _e1052_));
                        }
                        let _e1056_ = phi_2570_;
                        phi_2571_ = _e1056_;
                    }
                    let _e1058_ = phi_2571_;
                    phi_2572_ = _e1058_;
                } else {
                    phi_2572_ = 0f;
                }
                let _e1060_ = phi_2572_;
                let _e1061_ = (1f - _e790_);
                let _e1062_ = (1f - _e808_);
                let _e1063_ = (1f - _e826_);
                let _e1064_ = (1f - _e844_);
                if _e263_ {
                    let _e1065_ = (_e262_ > 0u);
                    if _e1065_ {
                        let _e1066_ = (_e262_ < _e255_);
                        let _e1067_ = (_e196_ == 0u);
                        if _e1067_ {
                            if _e1066_ {
                                let _e1070_ = (((_e232_ * _e255_) + _e262_) * 12u);
                                let _e1071_ = (_e1070_ < _e193_);
                                if _e1071_ {
                                    if _e1071_ {
                                    } else {
                                        break;
                                    }
                                    let _e1074_ = tile_checkpoints.member[_e1070_];
                                    phi_2607_ = _e1074_;
                                } else {
                                    phi_2607_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1076_ = phi_2607_;
                                let _e1078_ = bitcast<u32>(_e1076_.x);
                                let _e1079_ = (_e1078_ & 2139095040u);
                                if (_e1079_ == 2139095040u) {
                                    phi_2619_ = true;
                                } else {
                                    phi_2619_ = (_e1079_ == 0u);
                                }
                                let _e1083_ = phi_2619_;
                                let _e1087_ = bitcast<u32>(_e1076_.y);
                                let _e1088_ = (_e1087_ & 2139095040u);
                                if (_e1088_ == 2139095040u) {
                                    phi_2636_ = true;
                                } else {
                                    phi_2636_ = (_e1088_ == 0u);
                                }
                                let _e1092_ = phi_2636_;
                                let _e1096_ = bitcast<u32>(_e1076_.z);
                                let _e1097_ = (_e1096_ & 2139095040u);
                                if (_e1097_ == 2139095040u) {
                                    phi_2653_ = true;
                                } else {
                                    phi_2653_ = (_e1097_ == 0u);
                                }
                                let _e1101_ = phi_2653_;
                                let _e1105_ = bitcast<u32>(_e1076_.w);
                                let _e1106_ = (_e1105_ & 2139095040u);
                                if (_e1106_ == 2139095040u) {
                                    phi_2670_ = true;
                                } else {
                                    phi_2670_ = (_e1106_ == 0u);
                                }
                                let _e1110_ = phi_2670_;
                                local = array<f32, 4>(bitcast<f32>(select(_e1078_, 0u, _e1083_)), bitcast<f32>(select(_e1087_, 0u, _e1092_)), bitcast<f32>(select(_e1096_, 0u, _e1101_)), bitcast<f32>(select(_e1105_, 0u, _e1110_)));
                                let _e1114_ = (_e1070_ + 1u);
                                let _e1115_ = (_e1114_ < _e193_);
                                if _e1115_ {
                                    if _e1115_ {
                                    } else {
                                        break;
                                    }
                                    let _e1118_ = tile_checkpoints.member[_e1114_];
                                    phi_2694_ = _e1118_;
                                } else {
                                    phi_2694_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1120_ = phi_2694_;
                                let _e1122_ = bitcast<u32>(_e1120_.x);
                                let _e1123_ = (_e1122_ & 2139095040u);
                                if (_e1123_ == 2139095040u) {
                                    phi_2706_ = true;
                                } else {
                                    phi_2706_ = (_e1123_ == 0u);
                                }
                                let _e1127_ = phi_2706_;
                                let _e1131_ = bitcast<u32>(_e1120_.y);
                                let _e1132_ = (_e1131_ & 2139095040u);
                                if (_e1132_ == 2139095040u) {
                                    phi_2723_ = true;
                                } else {
                                    phi_2723_ = (_e1132_ == 0u);
                                }
                                let _e1136_ = phi_2723_;
                                let _e1140_ = bitcast<u32>(_e1120_.z);
                                let _e1141_ = (_e1140_ & 2139095040u);
                                if (_e1141_ == 2139095040u) {
                                    phi_2740_ = true;
                                } else {
                                    phi_2740_ = (_e1141_ == 0u);
                                }
                                let _e1145_ = phi_2740_;
                                let _e1149_ = bitcast<u32>(_e1120_.w);
                                let _e1150_ = (_e1149_ & 2139095040u);
                                if (_e1150_ == 2139095040u) {
                                    phi_2757_ = true;
                                } else {
                                    phi_2757_ = (_e1150_ == 0u);
                                }
                                let _e1154_ = phi_2757_;
                                local_1_ = array<f32, 4>(bitcast<f32>(select(_e1122_, 0u, _e1127_)), bitcast<f32>(select(_e1131_, 0u, _e1136_)), bitcast<f32>(select(_e1140_, 0u, _e1145_)), bitcast<f32>(select(_e1149_, 0u, _e1154_)));
                                let _e1158_ = (_e1070_ + 2u);
                                let _e1159_ = (_e1158_ < _e193_);
                                if _e1159_ {
                                    if _e1159_ {
                                    } else {
                                        break;
                                    }
                                    let _e1162_ = tile_checkpoints.member[_e1158_];
                                    phi_2781_ = _e1162_;
                                } else {
                                    phi_2781_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1164_ = phi_2781_;
                                let _e1166_ = bitcast<u32>(_e1164_.x);
                                let _e1167_ = (_e1166_ & 2139095040u);
                                if (_e1167_ == 2139095040u) {
                                    phi_2793_ = true;
                                } else {
                                    phi_2793_ = (_e1167_ == 0u);
                                }
                                let _e1171_ = phi_2793_;
                                let _e1175_ = bitcast<u32>(_e1164_.y);
                                let _e1176_ = (_e1175_ & 2139095040u);
                                if (_e1176_ == 2139095040u) {
                                    phi_2810_ = true;
                                } else {
                                    phi_2810_ = (_e1176_ == 0u);
                                }
                                let _e1180_ = phi_2810_;
                                let _e1184_ = bitcast<u32>(_e1164_.z);
                                let _e1185_ = (_e1184_ & 2139095040u);
                                if (_e1185_ == 2139095040u) {
                                    phi_2827_ = true;
                                } else {
                                    phi_2827_ = (_e1185_ == 0u);
                                }
                                let _e1189_ = phi_2827_;
                                let _e1193_ = bitcast<u32>(_e1164_.w);
                                let _e1194_ = (_e1193_ & 2139095040u);
                                if (_e1194_ == 2139095040u) {
                                    phi_2844_ = true;
                                } else {
                                    phi_2844_ = (_e1194_ == 0u);
                                }
                                let _e1198_ = phi_2844_;
                                local_2_ = array<f32, 4>(bitcast<f32>(select(_e1166_, 0u, _e1171_)), bitcast<f32>(select(_e1175_, 0u, _e1180_)), bitcast<f32>(select(_e1184_, 0u, _e1189_)), bitcast<f32>(select(_e1193_, 0u, _e1198_)));
                                let _e1202_ = (_e1070_ + 3u);
                                let _e1203_ = (_e1202_ < _e193_);
                                if _e1203_ {
                                    if _e1203_ {
                                    } else {
                                        break;
                                    }
                                    let _e1206_ = tile_checkpoints.member[_e1202_];
                                    phi_2868_ = _e1206_;
                                } else {
                                    phi_2868_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1208_ = phi_2868_;
                                let _e1210_ = bitcast<u32>(_e1208_.x);
                                let _e1211_ = (_e1210_ & 2139095040u);
                                if (_e1211_ == 2139095040u) {
                                    phi_2880_ = true;
                                } else {
                                    phi_2880_ = (_e1211_ == 0u);
                                }
                                let _e1215_ = phi_2880_;
                                let _e1219_ = bitcast<u32>(_e1208_.y);
                                let _e1220_ = (_e1219_ & 2139095040u);
                                if (_e1220_ == 2139095040u) {
                                    phi_2897_ = true;
                                } else {
                                    phi_2897_ = (_e1220_ == 0u);
                                }
                                let _e1224_ = phi_2897_;
                                let _e1228_ = bitcast<u32>(_e1208_.z);
                                let _e1229_ = (_e1228_ & 2139095040u);
                                if (_e1229_ == 2139095040u) {
                                    phi_2914_ = true;
                                } else {
                                    phi_2914_ = (_e1229_ == 0u);
                                }
                                let _e1233_ = phi_2914_;
                                let _e1237_ = bitcast<u32>(_e1208_.w);
                                let _e1238_ = (_e1237_ & 2139095040u);
                                if (_e1238_ == 2139095040u) {
                                    phi_2931_ = true;
                                } else {
                                    phi_2931_ = (_e1238_ == 0u);
                                }
                                let _e1242_ = phi_2931_;
                                local_3_ = array<f32, 4>(bitcast<f32>(select(_e1210_, 0u, _e1215_)), bitcast<f32>(select(_e1219_, 0u, _e1224_)), bitcast<f32>(select(_e1228_, 0u, _e1233_)), bitcast<f32>(select(_e1237_, 0u, _e1242_)));
                                let _e1246_ = (_e1070_ + 4u);
                                let _e1247_ = (_e1246_ < _e193_);
                                if _e1247_ {
                                    if _e1247_ {
                                    } else {
                                        break;
                                    }
                                    let _e1250_ = tile_checkpoints.member[_e1246_];
                                    phi_2955_ = _e1250_;
                                } else {
                                    phi_2955_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1252_ = phi_2955_;
                                let _e1254_ = bitcast<u32>(_e1252_.x);
                                let _e1255_ = (_e1254_ & 2139095040u);
                                if (_e1255_ == 2139095040u) {
                                    phi_2967_ = true;
                                } else {
                                    phi_2967_ = (_e1255_ == 0u);
                                }
                                let _e1259_ = phi_2967_;
                                let _e1263_ = bitcast<u32>(_e1252_.y);
                                let _e1264_ = (_e1263_ & 2139095040u);
                                if (_e1264_ == 2139095040u) {
                                    phi_2984_ = true;
                                } else {
                                    phi_2984_ = (_e1264_ == 0u);
                                }
                                let _e1268_ = phi_2984_;
                                let _e1272_ = bitcast<u32>(_e1252_.z);
                                let _e1273_ = (_e1272_ & 2139095040u);
                                if (_e1273_ == 2139095040u) {
                                    phi_3001_ = true;
                                } else {
                                    phi_3001_ = (_e1273_ == 0u);
                                }
                                let _e1277_ = phi_3001_;
                                let _e1281_ = bitcast<u32>(_e1252_.w);
                                let _e1282_ = (_e1281_ & 2139095040u);
                                if (_e1282_ == 2139095040u) {
                                    phi_3018_ = true;
                                } else {
                                    phi_3018_ = (_e1282_ == 0u);
                                }
                                let _e1286_ = phi_3018_;
                                local_4_ = array<f32, 4>(bitcast<f32>(select(_e1254_, 0u, _e1259_)), bitcast<f32>(select(_e1263_, 0u, _e1268_)), bitcast<f32>(select(_e1272_, 0u, _e1277_)), bitcast<f32>(select(_e1281_, 0u, _e1286_)));
                                let _e1290_ = (_e1070_ + 5u);
                                let _e1291_ = (_e1290_ < _e193_);
                                if _e1291_ {
                                    if _e1291_ {
                                    } else {
                                        break;
                                    }
                                    let _e1294_ = tile_checkpoints.member[_e1290_];
                                    phi_3042_ = _e1294_;
                                } else {
                                    phi_3042_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1296_ = phi_3042_;
                                let _e1298_ = bitcast<u32>(_e1296_.x);
                                let _e1299_ = (_e1298_ & 2139095040u);
                                if (_e1299_ == 2139095040u) {
                                    phi_3054_ = true;
                                } else {
                                    phi_3054_ = (_e1299_ == 0u);
                                }
                                let _e1303_ = phi_3054_;
                                let _e1307_ = bitcast<u32>(_e1296_.y);
                                let _e1308_ = (_e1307_ & 2139095040u);
                                if (_e1308_ == 2139095040u) {
                                    phi_3071_ = true;
                                } else {
                                    phi_3071_ = (_e1308_ == 0u);
                                }
                                let _e1312_ = phi_3071_;
                                let _e1316_ = bitcast<u32>(_e1296_.z);
                                let _e1317_ = (_e1316_ & 2139095040u);
                                if (_e1317_ == 2139095040u) {
                                    phi_3088_ = true;
                                } else {
                                    phi_3088_ = (_e1317_ == 0u);
                                }
                                let _e1321_ = phi_3088_;
                                let _e1325_ = bitcast<u32>(_e1296_.w);
                                let _e1326_ = (_e1325_ & 2139095040u);
                                if (_e1326_ == 2139095040u) {
                                    phi_3105_ = true;
                                } else {
                                    phi_3105_ = (_e1326_ == 0u);
                                }
                                let _e1330_ = phi_3105_;
                                local_5_ = array<f32, 4>(bitcast<f32>(select(_e1298_, 0u, _e1303_)), bitcast<f32>(select(_e1307_, 0u, _e1312_)), bitcast<f32>(select(_e1316_, 0u, _e1321_)), bitcast<f32>(select(_e1325_, 0u, _e1330_)));
                                let _e1334_ = (_e1070_ + 6u);
                                let _e1335_ = (_e1334_ < _e193_);
                                if _e1335_ {
                                    if _e1335_ {
                                    } else {
                                        break;
                                    }
                                    let _e1338_ = tile_checkpoints.member[_e1334_];
                                    phi_3129_ = _e1338_;
                                } else {
                                    phi_3129_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1340_ = phi_3129_;
                                let _e1346_ = select(0f, _e1340_.x, (bitcast<f32>((bitcast<u32>(_e1340_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1346_ != _e1346_) {
                                    phi_23974_ = true;
                                } else {
                                    phi_23974_ = (-4096f >= _e1346_);
                                }
                                let _e1350_ = phi_23974_;
                                let _e1351_ = select(_e1346_, -4096f, _e1350_);
                                if (_e1351_ != _e1351_) {
                                    phi_24034_ = true;
                                } else {
                                    phi_24034_ = (4096f <= _e1351_);
                                }
                                let _e1355_ = phi_24034_;
                                let _e1362_ = select(0f, _e1340_.y, (bitcast<f32>((bitcast<u32>(_e1340_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1362_ != _e1362_) {
                                    phi_24094_ = true;
                                } else {
                                    phi_24094_ = (-4096f >= _e1362_);
                                }
                                let _e1366_ = phi_24094_;
                                let _e1367_ = select(_e1362_, -4096f, _e1366_);
                                if (_e1367_ != _e1367_) {
                                    phi_24154_ = true;
                                } else {
                                    phi_24154_ = (4096f <= _e1367_);
                                }
                                let _e1371_ = phi_24154_;
                                let _e1378_ = select(0f, _e1340_.z, (bitcast<f32>((bitcast<u32>(_e1340_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1378_ != _e1378_) {
                                    phi_24214_ = true;
                                } else {
                                    phi_24214_ = (-4096f >= _e1378_);
                                }
                                let _e1382_ = phi_24214_;
                                let _e1383_ = select(_e1378_, -4096f, _e1382_);
                                if (_e1383_ != _e1383_) {
                                    phi_24274_ = true;
                                } else {
                                    phi_24274_ = (4096f <= _e1383_);
                                }
                                let _e1387_ = phi_24274_;
                                let _e1394_ = select(0f, _e1340_.w, (bitcast<f32>((bitcast<u32>(_e1340_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1394_ != _e1394_) {
                                    phi_24334_ = true;
                                } else {
                                    phi_24334_ = (-4096f >= _e1394_);
                                }
                                let _e1398_ = phi_24334_;
                                let _e1399_ = select(_e1394_, -4096f, _e1398_);
                                if (_e1399_ != _e1399_) {
                                    phi_24394_ = true;
                                } else {
                                    phi_24394_ = (4096f <= _e1399_);
                                }
                                let _e1403_ = phi_24394_;
                                local_6_ = array<f32, 4>(select(_e1351_, 4096f, _e1355_), select(_e1367_, 4096f, _e1371_), select(_e1383_, 4096f, _e1387_), select(_e1399_, 4096f, _e1403_));
                                let _e1406_ = (_e1070_ + 7u);
                                let _e1407_ = (_e1406_ < _e193_);
                                if _e1407_ {
                                    if _e1407_ {
                                    } else {
                                        break;
                                    }
                                    let _e1410_ = tile_checkpoints.member[_e1406_];
                                    phi_3320_ = _e1410_;
                                } else {
                                    phi_3320_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1412_ = phi_3320_;
                                let _e1418_ = select(0f, _e1412_.x, (bitcast<f32>((bitcast<u32>(_e1412_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1418_ != _e1418_) {
                                    phi_24454_ = true;
                                } else {
                                    phi_24454_ = (-4096f >= _e1418_);
                                }
                                let _e1422_ = phi_24454_;
                                let _e1423_ = select(_e1418_, -4096f, _e1422_);
                                if (_e1423_ != _e1423_) {
                                    phi_24514_ = true;
                                } else {
                                    phi_24514_ = (4096f <= _e1423_);
                                }
                                let _e1427_ = phi_24514_;
                                let _e1434_ = select(0f, _e1412_.y, (bitcast<f32>((bitcast<u32>(_e1412_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1434_ != _e1434_) {
                                    phi_24574_ = true;
                                } else {
                                    phi_24574_ = (-4096f >= _e1434_);
                                }
                                let _e1438_ = phi_24574_;
                                let _e1439_ = select(_e1434_, -4096f, _e1438_);
                                if (_e1439_ != _e1439_) {
                                    phi_24634_ = true;
                                } else {
                                    phi_24634_ = (4096f <= _e1439_);
                                }
                                let _e1443_ = phi_24634_;
                                let _e1450_ = select(0f, _e1412_.z, (bitcast<f32>((bitcast<u32>(_e1412_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1450_ != _e1450_) {
                                    phi_24694_ = true;
                                } else {
                                    phi_24694_ = (-4096f >= _e1450_);
                                }
                                let _e1454_ = phi_24694_;
                                let _e1455_ = select(_e1450_, -4096f, _e1454_);
                                if (_e1455_ != _e1455_) {
                                    phi_24754_ = true;
                                } else {
                                    phi_24754_ = (4096f <= _e1455_);
                                }
                                let _e1459_ = phi_24754_;
                                let _e1466_ = select(0f, _e1412_.w, (bitcast<f32>((bitcast<u32>(_e1412_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                if (_e1466_ != _e1466_) {
                                    phi_24814_ = true;
                                } else {
                                    phi_24814_ = (-4096f >= _e1466_);
                                }
                                let _e1470_ = phi_24814_;
                                let _e1471_ = select(_e1466_, -4096f, _e1470_);
                                if (_e1471_ != _e1471_) {
                                    phi_24874_ = true;
                                } else {
                                    phi_24874_ = (4096f <= _e1471_);
                                }
                                let _e1475_ = phi_24874_;
                                local_7_ = array<f32, 4>(select(_e1423_, 4096f, _e1427_), select(_e1439_, 4096f, _e1443_), select(_e1455_, 4096f, _e1459_), select(_e1471_, 4096f, _e1475_));
                                let _e1478_ = (_e1070_ + 8u);
                                let _e1479_ = (_e1478_ < _e193_);
                                if _e1479_ {
                                    if _e1479_ {
                                    } else {
                                        break;
                                    }
                                    let _e1482_ = tile_checkpoints.member[_e1478_];
                                    phi_3511_ = _e1482_;
                                } else {
                                    phi_3511_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1484_ = phi_3511_;
                                let _e1486_ = bitcast<u32>(_e1484_.x);
                                let _e1487_ = (_e1486_ & 2139095040u);
                                if (_e1487_ == 2139095040u) {
                                    phi_3523_ = true;
                                } else {
                                    phi_3523_ = (_e1487_ == 0u);
                                }
                                let _e1491_ = phi_3523_;
                                let _e1495_ = bitcast<u32>(_e1484_.y);
                                let _e1496_ = (_e1495_ & 2139095040u);
                                if (_e1496_ == 2139095040u) {
                                    phi_3540_ = true;
                                } else {
                                    phi_3540_ = (_e1496_ == 0u);
                                }
                                let _e1500_ = phi_3540_;
                                let _e1504_ = bitcast<u32>(_e1484_.z);
                                let _e1505_ = (_e1504_ & 2139095040u);
                                if (_e1505_ == 2139095040u) {
                                    phi_3557_ = true;
                                } else {
                                    phi_3557_ = (_e1505_ == 0u);
                                }
                                let _e1509_ = phi_3557_;
                                let _e1513_ = bitcast<u32>(_e1484_.w);
                                let _e1514_ = (_e1513_ & 2139095040u);
                                if (_e1514_ == 2139095040u) {
                                    phi_3574_ = true;
                                } else {
                                    phi_3574_ = (_e1514_ == 0u);
                                }
                                let _e1518_ = phi_3574_;
                                local_8_ = array<f32, 4>(bitcast<f32>(select(_e1486_, 0u, _e1491_)), bitcast<f32>(select(_e1495_, 0u, _e1500_)), bitcast<f32>(select(_e1504_, 0u, _e1509_)), bitcast<f32>(select(_e1513_, 0u, _e1518_)));
                                let _e1522_ = (_e1070_ + 9u);
                                let _e1523_ = (_e1522_ < _e193_);
                                if _e1523_ {
                                    if _e1523_ {
                                    } else {
                                        break;
                                    }
                                    let _e1526_ = tile_checkpoints.member[_e1522_];
                                    phi_3598_ = _e1526_;
                                } else {
                                    phi_3598_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1528_ = phi_3598_;
                                let _e1530_ = bitcast<u32>(_e1528_.x);
                                let _e1531_ = (_e1530_ & 2139095040u);
                                if (_e1531_ == 2139095040u) {
                                    phi_3610_ = true;
                                } else {
                                    phi_3610_ = (_e1531_ == 0u);
                                }
                                let _e1535_ = phi_3610_;
                                let _e1539_ = bitcast<u32>(_e1528_.y);
                                let _e1540_ = (_e1539_ & 2139095040u);
                                if (_e1540_ == 2139095040u) {
                                    phi_3627_ = true;
                                } else {
                                    phi_3627_ = (_e1540_ == 0u);
                                }
                                let _e1544_ = phi_3627_;
                                let _e1548_ = bitcast<u32>(_e1528_.z);
                                let _e1549_ = (_e1548_ & 2139095040u);
                                if (_e1549_ == 2139095040u) {
                                    phi_3644_ = true;
                                } else {
                                    phi_3644_ = (_e1549_ == 0u);
                                }
                                let _e1553_ = phi_3644_;
                                let _e1557_ = bitcast<u32>(_e1528_.w);
                                let _e1558_ = (_e1557_ & 2139095040u);
                                if (_e1558_ == 2139095040u) {
                                    phi_3661_ = true;
                                } else {
                                    phi_3661_ = (_e1558_ == 0u);
                                }
                                let _e1562_ = phi_3661_;
                                local_9_ = array<f32, 4>(bitcast<f32>(select(_e1530_, 0u, _e1535_)), bitcast<f32>(select(_e1539_, 0u, _e1544_)), bitcast<f32>(select(_e1548_, 0u, _e1553_)), bitcast<f32>(select(_e1557_, 0u, _e1562_)));
                                let _e1566_ = (_e1070_ + 10u);
                                let _e1567_ = (_e1566_ < _e193_);
                                if _e1567_ {
                                    if _e1567_ {
                                    } else {
                                        break;
                                    }
                                    let _e1570_ = tile_checkpoints.member[_e1566_];
                                    phi_3685_ = _e1570_;
                                } else {
                                    phi_3685_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1572_ = phi_3685_;
                                let _e1574_ = bitcast<u32>(_e1572_.x);
                                let _e1575_ = (_e1574_ & 2139095040u);
                                if (_e1575_ == 2139095040u) {
                                    phi_3697_ = true;
                                } else {
                                    phi_3697_ = (_e1575_ == 0u);
                                }
                                let _e1579_ = phi_3697_;
                                let _e1583_ = bitcast<u32>(_e1572_.y);
                                let _e1584_ = (_e1583_ & 2139095040u);
                                if (_e1584_ == 2139095040u) {
                                    phi_3714_ = true;
                                } else {
                                    phi_3714_ = (_e1584_ == 0u);
                                }
                                let _e1588_ = phi_3714_;
                                let _e1592_ = bitcast<u32>(_e1572_.z);
                                let _e1593_ = (_e1592_ & 2139095040u);
                                if (_e1593_ == 2139095040u) {
                                    phi_3731_ = true;
                                } else {
                                    phi_3731_ = (_e1593_ == 0u);
                                }
                                let _e1597_ = phi_3731_;
                                let _e1601_ = bitcast<u32>(_e1572_.w);
                                let _e1602_ = (_e1601_ & 2139095040u);
                                if (_e1602_ == 2139095040u) {
                                    phi_3748_ = true;
                                } else {
                                    phi_3748_ = (_e1602_ == 0u);
                                }
                                let _e1606_ = phi_3748_;
                                local_10_ = array<f32, 4>(bitcast<f32>(select(_e1574_, 0u, _e1579_)), bitcast<f32>(select(_e1583_, 0u, _e1588_)), bitcast<f32>(select(_e1592_, 0u, _e1597_)), bitcast<f32>(select(_e1601_, 0u, _e1606_)));
                                let _e1610_ = (_e1070_ + 11u);
                                let _e1611_ = (_e1610_ < _e193_);
                                if _e1611_ {
                                    if _e1611_ {
                                    } else {
                                        break;
                                    }
                                    let _e1614_ = tile_checkpoints.member[_e1610_];
                                    phi_3772_ = _e1614_;
                                } else {
                                    phi_3772_ = vec4<f32>(0f, 0f, 0f, 0f);
                                }
                                let _e1616_ = phi_3772_;
                                let _e1618_ = bitcast<u32>(_e1616_.x);
                                let _e1619_ = (_e1618_ & 2139095040u);
                                if (_e1619_ == 2139095040u) {
                                    phi_3784_ = true;
                                } else {
                                    phi_3784_ = (_e1619_ == 0u);
                                }
                                let _e1623_ = phi_3784_;
                                let _e1627_ = bitcast<u32>(_e1616_.y);
                                let _e1628_ = (_e1627_ & 2139095040u);
                                if (_e1628_ == 2139095040u) {
                                    phi_3801_ = true;
                                } else {
                                    phi_3801_ = (_e1628_ == 0u);
                                }
                                let _e1632_ = phi_3801_;
                                let _e1636_ = bitcast<u32>(_e1616_.z);
                                let _e1637_ = (_e1636_ & 2139095040u);
                                if (_e1637_ == 2139095040u) {
                                    phi_3818_ = true;
                                } else {
                                    phi_3818_ = (_e1637_ == 0u);
                                }
                                let _e1641_ = phi_3818_;
                                let _e1645_ = bitcast<u32>(_e1616_.w);
                                let _e1646_ = (_e1645_ & 2139095040u);
                                if (_e1646_ == 2139095040u) {
                                    phi_3835_ = true;
                                } else {
                                    phi_3835_ = (_e1646_ == 0u);
                                }
                                let _e1650_ = phi_3835_;
                                local_11_ = array<f32, 4>(bitcast<f32>(select(_e1618_, 0u, _e1623_)), bitcast<f32>(select(_e1627_, 0u, _e1632_)), bitcast<f32>(select(_e1636_, 0u, _e1641_)), bitcast<f32>(select(_e1645_, 0u, _e1650_)));
                                phi_3849_ = type_15_(0u, 4u);
                                loop {
                                    let _e1655_ = phi_3849_;
                                    if (_e1655_.member < _e1655_.member_1_) {
                                        phi_3850_ = type_15_((_e1655_.member + 1u), _e1655_.member_1_);
                                        phi_3873_ = type_15_(1u, _e1655_.member);
                                    } else {
                                        phi_3850_ = _e1655_;
                                        phi_3873_ = type_15_(0u, u32());
                                    }
                                    let _e1668_ = phi_3850_;
                                    let _e1670_ = phi_3873_;
                                    let _e1674_ = (bitcast<i32>(_e1670_.member) != 0i);
                                    if _e1674_ {
                                        let _e1675_ = (_e1670_.member_1_ < 4u);
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1677_ = local[_e1670_.member_1_];
                                        if (_e1670_.member_1_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1670_.member_1_] = _e1677_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1681_ = local_2_[_e1670_.member_1_];
                                        let _e1682_ = (4u + _e1670_.member_1_);
                                        if (_e1682_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1682_] = _e1681_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1686_ = local_1_[_e1670_.member_1_];
                                        let _e1687_ = (8u + _e1670_.member_1_);
                                        if (_e1687_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1687_] = _e1686_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1691_ = local_3_[_e1670_.member_1_];
                                        let _e1692_ = (12u + _e1670_.member_1_);
                                        if (_e1692_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1692_] = _e1691_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1696_ = local_4_[_e1670_.member_1_];
                                        let _e1697_ = (16u + _e1670_.member_1_);
                                        if (_e1697_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1697_] = _e1696_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1701_ = local_5_[_e1670_.member_1_];
                                        let _e1702_ = (20u + _e1670_.member_1_);
                                        if (_e1702_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1702_] = _e1701_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1706_ = local_6_[_e1670_.member_1_];
                                        let _e1707_ = (24u + _e1670_.member_1_);
                                        if (_e1707_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1707_] = _e1706_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1711_ = local_7_[_e1670_.member_1_];
                                        let _e1712_ = (28u + _e1670_.member_1_);
                                        if (_e1712_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1712_] = _e1711_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1716_ = local_8_[_e1670_.member_1_];
                                        let _e1717_ = (32u + _e1670_.member_1_);
                                        if (_e1717_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1717_] = _e1716_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1721_ = local_10_[_e1670_.member_1_];
                                        let _e1722_ = (36u + _e1670_.member_1_);
                                        if (_e1722_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1722_] = _e1721_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1726_ = local_9_[_e1670_.member_1_];
                                        let _e1727_ = (40u + _e1670_.member_1_);
                                        if (_e1727_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1727_] = _e1726_;
                                        if _e1675_ {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        let _e1731_ = local_11_[_e1670_.member_1_];
                                        let _e1732_ = (44u + _e1670_.member_1_);
                                        if (_e1732_ < 256u) {
                                        } else {
                                            phi_35544_ = true;
                                            break;
                                        }
                                        aux_scratch[_e1732_] = _e1731_;
                                    }
                                    phi_3849_ = _e1668_;
                                    phi_35544_ = false;
                                    if !(_e1674_) {
                                        break;
                                    }
                                }
                                let _e1737_ = phi_35544_;
                                if _e1737_ {
                                    break;
                                }
                                phi_35979_ = _e1737_;
                            } else {
                                phi_35979_ = false;
                            }
                            let _e1739_ = phi_35979_;
                            phi_35978_ = _e1739_;
                            phi_4049_ = select(true, false, _e1066_);
                        } else {
                            phi_35978_ = false;
                            phi_4049_ = false;
                        }
                        let _e1742_ = phi_35978_;
                        let _e1744_ = phi_4049_;
                        phi_35977_ = _e1742_;
                        phi_4051_ = _e1066_;
                        phi_4055_ = _e1744_;
                        phi_4056_ = select(false, _e1066_, _e1067_);
                    } else {
                        phi_35977_ = false;
                        phi_4051_ = bool();
                        phi_4055_ = false;
                        phi_4056_ = false;
                    }
                    let _e1747_ = phi_35977_;
                    let _e1749_ = phi_4051_;
                    let _e1751_ = phi_4055_;
                    let _e1753_ = phi_4056_;
                    phi_35976_ = _e1747_;
                    phi_4057_ = _e1749_;
                    phi_4060_ = select(true, false, _e1065_);
                    phi_4061_ = _e1751_;
                    phi_4062_ = _e1753_;
                } else {
                    phi_35976_ = false;
                    phi_4057_ = bool();
                    phi_4060_ = true;
                    phi_4061_ = false;
                    phi_4062_ = false;
                }
                let _e1756_ = phi_35976_;
                let _e1758_ = phi_4057_;
                let _e1760_ = phi_4060_;
                let _e1762_ = phi_4061_;
                let _e1764_ = phi_4062_;
                if _e1760_ {
                    phi_4072_ = (_e196_ == 0u);
                } else {
                    phi_4072_ = _e1762_;
                }
                let _e1767_ = phi_4072_;
                let _e1768_ = select(_e1758_, false, _e1760_);
                if _e1767_ {
                    phi_4079_ = type_15_(0u, 48u);
                    loop {
                        let _e1771_ = phi_4079_;
                        if (_e1771_.member < _e1771_.member_1_) {
                            phi_4080_ = type_15_((_e1771_.member + 1u), _e1771_.member_1_);
                            phi_4103_ = type_15_(1u, _e1771_.member);
                        } else {
                            phi_4080_ = _e1771_;
                            phi_4103_ = type_15_(0u, u32());
                        }
                        let _e1784_ = phi_4080_;
                        let _e1786_ = phi_4103_;
                        let _e1790_ = (bitcast<i32>(_e1786_.member) != 0i);
                        if _e1790_ {
                            if (_e1786_.member_1_ < 256u) {
                            } else {
                                phi_35971_ = true;
                                break;
                            }
                            aux_scratch[_e1786_.member_1_] = 0f;
                        }
                        phi_4079_ = _e1784_;
                        phi_35971_ = _e1756_;
                        if !(_e1790_) {
                            break;
                        }
                    }
                    let _e1795_ = phi_35971_;
                    if _e1795_ {
                        break;
                    }
                    phi_36012_ = _e1795_;
                } else {
                    phi_36012_ = _e1756_;
                }
                let _e1797_ = phi_36012_;
                workgroupBarrier();
                phi_36009_ = _e1797_;
                phi_4129_ = type_15_(_e330_, _e332_);
                loop {
                    let _e1804_ = phi_36009_;
                    let _e1806_ = phi_4129_;
                    if (_e1806_.member < _e1806_.member_1_) {
                        phi_4130_ = type_15_((_e1806_.member + 1u), _e1806_.member_1_);
                        phi_4153_ = type_15_(1u, _e1806_.member);
                    } else {
                        phi_4130_ = _e1806_;
                        phi_4153_ = type_15_(0u, u32());
                    }
                    let _e1819_ = phi_4130_;
                    let _e1821_ = phi_4153_;
                    let _e1825_ = (bitcast<i32>(_e1821_.member) != 0i);
                    if _e1825_ {
                        let _e1827_ = (_e196_ == 0u);
                        if _e1827_ {
                            if select(select(_e1758_, _e1768_, _e1767_), _e1768_, select(true, false, select(select(_e1764_, false, _e1760_), true, _e1767_))) {
                                phi_4174_ = select(true, false, (_e1821_.member_1_ == _e262_));
                            } else {
                                phi_4174_ = true;
                            }
                            let _e1831_ = phi_4174_;
                            if _e1831_ {
                                let _e1834_ = (((_e232_ * _e255_) + _e1821_.member_1_) * 12u);
                                let _e1836_ = aux_scratch[0];
                                let _e1838_ = aux_scratch[1];
                                let _e1840_ = aux_scratch[2];
                                let _e1842_ = aux_scratch[3];
                                let _e1843_ = bitcast<u32>(_e1836_);
                                let _e1844_ = (_e1843_ & 2139095040u);
                                if (_e1844_ == 2139095040u) {
                                    phi_4222_ = true;
                                } else {
                                    phi_4222_ = (_e1844_ == 0u);
                                }
                                let _e1848_ = phi_4222_;
                                let _e1851_ = bitcast<u32>(_e1838_);
                                let _e1852_ = (_e1851_ & 2139095040u);
                                if (_e1852_ == 2139095040u) {
                                    phi_4238_ = true;
                                } else {
                                    phi_4238_ = (_e1852_ == 0u);
                                }
                                let _e1856_ = phi_4238_;
                                let _e1859_ = bitcast<u32>(_e1840_);
                                let _e1860_ = (_e1859_ & 2139095040u);
                                if (_e1860_ == 2139095040u) {
                                    phi_4254_ = true;
                                } else {
                                    phi_4254_ = (_e1860_ == 0u);
                                }
                                let _e1864_ = phi_4254_;
                                let _e1867_ = bitcast<u32>(_e1842_);
                                let _e1868_ = (_e1867_ & 2139095040u);
                                if (_e1868_ == 2139095040u) {
                                    phi_4270_ = true;
                                } else {
                                    phi_4270_ = (_e1868_ == 0u);
                                }
                                let _e1872_ = phi_4270_;
                                let _e1876_ = (_e1834_ < _e193_);
                                if _e1876_ {
                                    if _e1876_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1834_] = vec4<f32>(bitcast<f32>(select(_e1843_, 0u, _e1848_)), bitcast<f32>(select(_e1851_, 0u, _e1856_)), bitcast<f32>(select(_e1859_, 0u, _e1864_)), bitcast<f32>(select(_e1867_, 0u, _e1872_)));
                                }
                                let _e1879_ = (_e1834_ + 1u);
                                let _e1881_ = aux_scratch[8];
                                let _e1883_ = aux_scratch[9];
                                let _e1885_ = aux_scratch[10];
                                let _e1887_ = aux_scratch[11];
                                let _e1888_ = bitcast<u32>(_e1881_);
                                let _e1889_ = (_e1888_ & 2139095040u);
                                if (_e1889_ == 2139095040u) {
                                    phi_4329_ = true;
                                } else {
                                    phi_4329_ = (_e1889_ == 0u);
                                }
                                let _e1893_ = phi_4329_;
                                let _e1896_ = bitcast<u32>(_e1883_);
                                let _e1897_ = (_e1896_ & 2139095040u);
                                if (_e1897_ == 2139095040u) {
                                    phi_4345_ = true;
                                } else {
                                    phi_4345_ = (_e1897_ == 0u);
                                }
                                let _e1901_ = phi_4345_;
                                let _e1904_ = bitcast<u32>(_e1885_);
                                let _e1905_ = (_e1904_ & 2139095040u);
                                if (_e1905_ == 2139095040u) {
                                    phi_4361_ = true;
                                } else {
                                    phi_4361_ = (_e1905_ == 0u);
                                }
                                let _e1909_ = phi_4361_;
                                let _e1912_ = bitcast<u32>(_e1887_);
                                let _e1913_ = (_e1912_ & 2139095040u);
                                if (_e1913_ == 2139095040u) {
                                    phi_4377_ = true;
                                } else {
                                    phi_4377_ = (_e1913_ == 0u);
                                }
                                let _e1917_ = phi_4377_;
                                let _e1921_ = (_e1879_ < _e193_);
                                if _e1921_ {
                                    if _e1921_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1879_] = vec4<f32>(bitcast<f32>(select(_e1888_, 0u, _e1893_)), bitcast<f32>(select(_e1896_, 0u, _e1901_)), bitcast<f32>(select(_e1904_, 0u, _e1909_)), bitcast<f32>(select(_e1912_, 0u, _e1917_)));
                                }
                                let _e1924_ = (_e1834_ + 2u);
                                let _e1926_ = aux_scratch[4];
                                let _e1928_ = aux_scratch[5];
                                let _e1930_ = aux_scratch[6];
                                let _e1932_ = aux_scratch[7];
                                let _e1933_ = bitcast<u32>(_e1926_);
                                let _e1934_ = (_e1933_ & 2139095040u);
                                if (_e1934_ == 2139095040u) {
                                    phi_4436_ = true;
                                } else {
                                    phi_4436_ = (_e1934_ == 0u);
                                }
                                let _e1938_ = phi_4436_;
                                let _e1941_ = bitcast<u32>(_e1928_);
                                let _e1942_ = (_e1941_ & 2139095040u);
                                if (_e1942_ == 2139095040u) {
                                    phi_4452_ = true;
                                } else {
                                    phi_4452_ = (_e1942_ == 0u);
                                }
                                let _e1946_ = phi_4452_;
                                let _e1949_ = bitcast<u32>(_e1930_);
                                let _e1950_ = (_e1949_ & 2139095040u);
                                if (_e1950_ == 2139095040u) {
                                    phi_4468_ = true;
                                } else {
                                    phi_4468_ = (_e1950_ == 0u);
                                }
                                let _e1954_ = phi_4468_;
                                let _e1957_ = bitcast<u32>(_e1932_);
                                let _e1958_ = (_e1957_ & 2139095040u);
                                if (_e1958_ == 2139095040u) {
                                    phi_4484_ = true;
                                } else {
                                    phi_4484_ = (_e1958_ == 0u);
                                }
                                let _e1962_ = phi_4484_;
                                let _e1966_ = (_e1924_ < _e193_);
                                if _e1966_ {
                                    if _e1966_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1924_] = vec4<f32>(bitcast<f32>(select(_e1933_, 0u, _e1938_)), bitcast<f32>(select(_e1941_, 0u, _e1946_)), bitcast<f32>(select(_e1949_, 0u, _e1954_)), bitcast<f32>(select(_e1957_, 0u, _e1962_)));
                                }
                                let _e1969_ = (_e1834_ + 3u);
                                let _e1971_ = aux_scratch[12];
                                let _e1973_ = aux_scratch[13];
                                let _e1975_ = aux_scratch[14];
                                let _e1977_ = aux_scratch[15];
                                let _e1978_ = bitcast<u32>(_e1971_);
                                let _e1979_ = (_e1978_ & 2139095040u);
                                if (_e1979_ == 2139095040u) {
                                    phi_4543_ = true;
                                } else {
                                    phi_4543_ = (_e1979_ == 0u);
                                }
                                let _e1983_ = phi_4543_;
                                let _e1986_ = bitcast<u32>(_e1973_);
                                let _e1987_ = (_e1986_ & 2139095040u);
                                if (_e1987_ == 2139095040u) {
                                    phi_4559_ = true;
                                } else {
                                    phi_4559_ = (_e1987_ == 0u);
                                }
                                let _e1991_ = phi_4559_;
                                let _e1994_ = bitcast<u32>(_e1975_);
                                let _e1995_ = (_e1994_ & 2139095040u);
                                if (_e1995_ == 2139095040u) {
                                    phi_4575_ = true;
                                } else {
                                    phi_4575_ = (_e1995_ == 0u);
                                }
                                let _e1999_ = phi_4575_;
                                let _e2002_ = bitcast<u32>(_e1977_);
                                let _e2003_ = (_e2002_ & 2139095040u);
                                if (_e2003_ == 2139095040u) {
                                    phi_4591_ = true;
                                } else {
                                    phi_4591_ = (_e2003_ == 0u);
                                }
                                let _e2007_ = phi_4591_;
                                let _e2011_ = (_e1969_ < _e193_);
                                if _e2011_ {
                                    if _e2011_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e1969_] = vec4<f32>(bitcast<f32>(select(_e1978_, 0u, _e1983_)), bitcast<f32>(select(_e1986_, 0u, _e1991_)), bitcast<f32>(select(_e1994_, 0u, _e1999_)), bitcast<f32>(select(_e2002_, 0u, _e2007_)));
                                }
                                let _e2014_ = (_e1834_ + 4u);
                                let _e2016_ = aux_scratch[16];
                                let _e2018_ = aux_scratch[17];
                                let _e2020_ = aux_scratch[18];
                                let _e2022_ = aux_scratch[19];
                                let _e2023_ = bitcast<u32>(_e2016_);
                                let _e2024_ = (_e2023_ & 2139095040u);
                                if (_e2024_ == 2139095040u) {
                                    phi_4650_ = true;
                                } else {
                                    phi_4650_ = (_e2024_ == 0u);
                                }
                                let _e2028_ = phi_4650_;
                                let _e2031_ = bitcast<u32>(_e2018_);
                                let _e2032_ = (_e2031_ & 2139095040u);
                                if (_e2032_ == 2139095040u) {
                                    phi_4666_ = true;
                                } else {
                                    phi_4666_ = (_e2032_ == 0u);
                                }
                                let _e2036_ = phi_4666_;
                                let _e2039_ = bitcast<u32>(_e2020_);
                                let _e2040_ = (_e2039_ & 2139095040u);
                                if (_e2040_ == 2139095040u) {
                                    phi_4682_ = true;
                                } else {
                                    phi_4682_ = (_e2040_ == 0u);
                                }
                                let _e2044_ = phi_4682_;
                                let _e2047_ = bitcast<u32>(_e2022_);
                                let _e2048_ = (_e2047_ & 2139095040u);
                                if (_e2048_ == 2139095040u) {
                                    phi_4698_ = true;
                                } else {
                                    phi_4698_ = (_e2048_ == 0u);
                                }
                                let _e2052_ = phi_4698_;
                                let _e2056_ = (_e2014_ < _e193_);
                                if _e2056_ {
                                    if _e2056_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2014_] = vec4<f32>(bitcast<f32>(select(_e2023_, 0u, _e2028_)), bitcast<f32>(select(_e2031_, 0u, _e2036_)), bitcast<f32>(select(_e2039_, 0u, _e2044_)), bitcast<f32>(select(_e2047_, 0u, _e2052_)));
                                }
                                let _e2059_ = (_e1834_ + 5u);
                                let _e2061_ = aux_scratch[20];
                                let _e2063_ = aux_scratch[21];
                                let _e2065_ = aux_scratch[22];
                                let _e2067_ = aux_scratch[23];
                                let _e2068_ = bitcast<u32>(_e2061_);
                                let _e2069_ = (_e2068_ & 2139095040u);
                                if (_e2069_ == 2139095040u) {
                                    phi_4757_ = true;
                                } else {
                                    phi_4757_ = (_e2069_ == 0u);
                                }
                                let _e2073_ = phi_4757_;
                                let _e2076_ = bitcast<u32>(_e2063_);
                                let _e2077_ = (_e2076_ & 2139095040u);
                                if (_e2077_ == 2139095040u) {
                                    phi_4773_ = true;
                                } else {
                                    phi_4773_ = (_e2077_ == 0u);
                                }
                                let _e2081_ = phi_4773_;
                                let _e2084_ = bitcast<u32>(_e2065_);
                                let _e2085_ = (_e2084_ & 2139095040u);
                                if (_e2085_ == 2139095040u) {
                                    phi_4789_ = true;
                                } else {
                                    phi_4789_ = (_e2085_ == 0u);
                                }
                                let _e2089_ = phi_4789_;
                                let _e2092_ = bitcast<u32>(_e2067_);
                                let _e2093_ = (_e2092_ & 2139095040u);
                                if (_e2093_ == 2139095040u) {
                                    phi_4805_ = true;
                                } else {
                                    phi_4805_ = (_e2093_ == 0u);
                                }
                                let _e2097_ = phi_4805_;
                                let _e2101_ = (_e2059_ < _e193_);
                                if _e2101_ {
                                    if _e2101_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2059_] = vec4<f32>(bitcast<f32>(select(_e2068_, 0u, _e2073_)), bitcast<f32>(select(_e2076_, 0u, _e2081_)), bitcast<f32>(select(_e2084_, 0u, _e2089_)), bitcast<f32>(select(_e2092_, 0u, _e2097_)));
                                }
                                let _e2104_ = (_e1834_ + 6u);
                                let _e2106_ = aux_scratch[24];
                                let _e2108_ = aux_scratch[25];
                                let _e2110_ = aux_scratch[26];
                                let _e2112_ = aux_scratch[27];
                                let _e2113_ = bitcast<u32>(_e2106_);
                                let _e2114_ = (_e2113_ & 2139095040u);
                                if (_e2114_ == 2139095040u) {
                                    phi_4864_ = true;
                                } else {
                                    phi_4864_ = (_e2114_ == 0u);
                                }
                                let _e2118_ = phi_4864_;
                                let _e2121_ = bitcast<u32>(_e2108_);
                                let _e2122_ = (_e2121_ & 2139095040u);
                                if (_e2122_ == 2139095040u) {
                                    phi_4880_ = true;
                                } else {
                                    phi_4880_ = (_e2122_ == 0u);
                                }
                                let _e2126_ = phi_4880_;
                                let _e2129_ = bitcast<u32>(_e2110_);
                                let _e2130_ = (_e2129_ & 2139095040u);
                                if (_e2130_ == 2139095040u) {
                                    phi_4896_ = true;
                                } else {
                                    phi_4896_ = (_e2130_ == 0u);
                                }
                                let _e2134_ = phi_4896_;
                                let _e2137_ = bitcast<u32>(_e2112_);
                                let _e2138_ = (_e2137_ & 2139095040u);
                                if (_e2138_ == 2139095040u) {
                                    phi_4912_ = true;
                                } else {
                                    phi_4912_ = (_e2138_ == 0u);
                                }
                                let _e2142_ = phi_4912_;
                                let _e2146_ = (_e2104_ < _e193_);
                                if _e2146_ {
                                    if _e2146_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2104_] = vec4<f32>(bitcast<f32>(select(_e2113_, 0u, _e2118_)), bitcast<f32>(select(_e2121_, 0u, _e2126_)), bitcast<f32>(select(_e2129_, 0u, _e2134_)), bitcast<f32>(select(_e2137_, 0u, _e2142_)));
                                }
                                let _e2149_ = (_e1834_ + 7u);
                                let _e2151_ = aux_scratch[28];
                                let _e2153_ = aux_scratch[29];
                                let _e2155_ = aux_scratch[30];
                                let _e2157_ = aux_scratch[31];
                                let _e2158_ = bitcast<u32>(_e2151_);
                                let _e2159_ = (_e2158_ & 2139095040u);
                                if (_e2159_ == 2139095040u) {
                                    phi_4971_ = true;
                                } else {
                                    phi_4971_ = (_e2159_ == 0u);
                                }
                                let _e2163_ = phi_4971_;
                                let _e2166_ = bitcast<u32>(_e2153_);
                                let _e2167_ = (_e2166_ & 2139095040u);
                                if (_e2167_ == 2139095040u) {
                                    phi_4987_ = true;
                                } else {
                                    phi_4987_ = (_e2167_ == 0u);
                                }
                                let _e2171_ = phi_4987_;
                                let _e2174_ = bitcast<u32>(_e2155_);
                                let _e2175_ = (_e2174_ & 2139095040u);
                                if (_e2175_ == 2139095040u) {
                                    phi_5003_ = true;
                                } else {
                                    phi_5003_ = (_e2175_ == 0u);
                                }
                                let _e2179_ = phi_5003_;
                                let _e2182_ = bitcast<u32>(_e2157_);
                                let _e2183_ = (_e2182_ & 2139095040u);
                                if (_e2183_ == 2139095040u) {
                                    phi_5019_ = true;
                                } else {
                                    phi_5019_ = (_e2183_ == 0u);
                                }
                                let _e2187_ = phi_5019_;
                                let _e2191_ = (_e2149_ < _e193_);
                                if _e2191_ {
                                    if _e2191_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2149_] = vec4<f32>(bitcast<f32>(select(_e2158_, 0u, _e2163_)), bitcast<f32>(select(_e2166_, 0u, _e2171_)), bitcast<f32>(select(_e2174_, 0u, _e2179_)), bitcast<f32>(select(_e2182_, 0u, _e2187_)));
                                }
                                let _e2194_ = (_e1834_ + 8u);
                                let _e2196_ = aux_scratch[32];
                                let _e2198_ = aux_scratch[33];
                                let _e2200_ = aux_scratch[34];
                                let _e2202_ = aux_scratch[35];
                                let _e2203_ = bitcast<u32>(_e2196_);
                                let _e2204_ = (_e2203_ & 2139095040u);
                                if (_e2204_ == 2139095040u) {
                                    phi_5078_ = true;
                                } else {
                                    phi_5078_ = (_e2204_ == 0u);
                                }
                                let _e2208_ = phi_5078_;
                                let _e2211_ = bitcast<u32>(_e2198_);
                                let _e2212_ = (_e2211_ & 2139095040u);
                                if (_e2212_ == 2139095040u) {
                                    phi_5094_ = true;
                                } else {
                                    phi_5094_ = (_e2212_ == 0u);
                                }
                                let _e2216_ = phi_5094_;
                                let _e2219_ = bitcast<u32>(_e2200_);
                                let _e2220_ = (_e2219_ & 2139095040u);
                                if (_e2220_ == 2139095040u) {
                                    phi_5110_ = true;
                                } else {
                                    phi_5110_ = (_e2220_ == 0u);
                                }
                                let _e2224_ = phi_5110_;
                                let _e2227_ = bitcast<u32>(_e2202_);
                                let _e2228_ = (_e2227_ & 2139095040u);
                                if (_e2228_ == 2139095040u) {
                                    phi_5126_ = true;
                                } else {
                                    phi_5126_ = (_e2228_ == 0u);
                                }
                                let _e2232_ = phi_5126_;
                                let _e2236_ = (_e2194_ < _e193_);
                                if _e2236_ {
                                    if _e2236_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2194_] = vec4<f32>(bitcast<f32>(select(_e2203_, 0u, _e2208_)), bitcast<f32>(select(_e2211_, 0u, _e2216_)), bitcast<f32>(select(_e2219_, 0u, _e2224_)), bitcast<f32>(select(_e2227_, 0u, _e2232_)));
                                }
                                let _e2239_ = (_e1834_ + 9u);
                                let _e2241_ = aux_scratch[40];
                                let _e2243_ = aux_scratch[41];
                                let _e2245_ = aux_scratch[42];
                                let _e2247_ = aux_scratch[43];
                                let _e2248_ = bitcast<u32>(_e2241_);
                                let _e2249_ = (_e2248_ & 2139095040u);
                                if (_e2249_ == 2139095040u) {
                                    phi_5185_ = true;
                                } else {
                                    phi_5185_ = (_e2249_ == 0u);
                                }
                                let _e2253_ = phi_5185_;
                                let _e2256_ = bitcast<u32>(_e2243_);
                                let _e2257_ = (_e2256_ & 2139095040u);
                                if (_e2257_ == 2139095040u) {
                                    phi_5201_ = true;
                                } else {
                                    phi_5201_ = (_e2257_ == 0u);
                                }
                                let _e2261_ = phi_5201_;
                                let _e2264_ = bitcast<u32>(_e2245_);
                                let _e2265_ = (_e2264_ & 2139095040u);
                                if (_e2265_ == 2139095040u) {
                                    phi_5217_ = true;
                                } else {
                                    phi_5217_ = (_e2265_ == 0u);
                                }
                                let _e2269_ = phi_5217_;
                                let _e2272_ = bitcast<u32>(_e2247_);
                                let _e2273_ = (_e2272_ & 2139095040u);
                                if (_e2273_ == 2139095040u) {
                                    phi_5233_ = true;
                                } else {
                                    phi_5233_ = (_e2273_ == 0u);
                                }
                                let _e2277_ = phi_5233_;
                                let _e2281_ = (_e2239_ < _e193_);
                                if _e2281_ {
                                    if _e2281_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2239_] = vec4<f32>(bitcast<f32>(select(_e2248_, 0u, _e2253_)), bitcast<f32>(select(_e2256_, 0u, _e2261_)), bitcast<f32>(select(_e2264_, 0u, _e2269_)), bitcast<f32>(select(_e2272_, 0u, _e2277_)));
                                }
                                let _e2284_ = (_e1834_ + 10u);
                                let _e2286_ = aux_scratch[36];
                                let _e2288_ = aux_scratch[37];
                                let _e2290_ = aux_scratch[38];
                                let _e2292_ = aux_scratch[39];
                                let _e2293_ = bitcast<u32>(_e2286_);
                                let _e2294_ = (_e2293_ & 2139095040u);
                                if (_e2294_ == 2139095040u) {
                                    phi_5292_ = true;
                                } else {
                                    phi_5292_ = (_e2294_ == 0u);
                                }
                                let _e2298_ = phi_5292_;
                                let _e2301_ = bitcast<u32>(_e2288_);
                                let _e2302_ = (_e2301_ & 2139095040u);
                                if (_e2302_ == 2139095040u) {
                                    phi_5308_ = true;
                                } else {
                                    phi_5308_ = (_e2302_ == 0u);
                                }
                                let _e2306_ = phi_5308_;
                                let _e2309_ = bitcast<u32>(_e2290_);
                                let _e2310_ = (_e2309_ & 2139095040u);
                                if (_e2310_ == 2139095040u) {
                                    phi_5324_ = true;
                                } else {
                                    phi_5324_ = (_e2310_ == 0u);
                                }
                                let _e2314_ = phi_5324_;
                                let _e2317_ = bitcast<u32>(_e2292_);
                                let _e2318_ = (_e2317_ & 2139095040u);
                                if (_e2318_ == 2139095040u) {
                                    phi_5340_ = true;
                                } else {
                                    phi_5340_ = (_e2318_ == 0u);
                                }
                                let _e2322_ = phi_5340_;
                                let _e2326_ = (_e2284_ < _e193_);
                                if _e2326_ {
                                    if _e2326_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2284_] = vec4<f32>(bitcast<f32>(select(_e2293_, 0u, _e2298_)), bitcast<f32>(select(_e2301_, 0u, _e2306_)), bitcast<f32>(select(_e2309_, 0u, _e2314_)), bitcast<f32>(select(_e2317_, 0u, _e2322_)));
                                }
                                let _e2329_ = (_e1834_ + 11u);
                                let _e2331_ = aux_scratch[44];
                                let _e2333_ = aux_scratch[45];
                                let _e2335_ = aux_scratch[46];
                                let _e2337_ = aux_scratch[47];
                                let _e2338_ = bitcast<u32>(_e2331_);
                                let _e2339_ = (_e2338_ & 2139095040u);
                                if (_e2339_ == 2139095040u) {
                                    phi_5399_ = true;
                                } else {
                                    phi_5399_ = (_e2339_ == 0u);
                                }
                                let _e2343_ = phi_5399_;
                                let _e2346_ = bitcast<u32>(_e2333_);
                                let _e2347_ = (_e2346_ & 2139095040u);
                                if (_e2347_ == 2139095040u) {
                                    phi_5415_ = true;
                                } else {
                                    phi_5415_ = (_e2347_ == 0u);
                                }
                                let _e2351_ = phi_5415_;
                                let _e2354_ = bitcast<u32>(_e2335_);
                                let _e2355_ = (_e2354_ & 2139095040u);
                                if (_e2355_ == 2139095040u) {
                                    phi_5431_ = true;
                                } else {
                                    phi_5431_ = (_e2355_ == 0u);
                                }
                                let _e2359_ = phi_5431_;
                                let _e2362_ = bitcast<u32>(_e2337_);
                                let _e2363_ = (_e2362_ & 2139095040u);
                                if (_e2363_ == 2139095040u) {
                                    phi_5447_ = true;
                                } else {
                                    phi_5447_ = (_e2363_ == 0u);
                                }
                                let _e2367_ = phi_5447_;
                                let _e2371_ = (_e2329_ < _e193_);
                                if _e2371_ {
                                    if _e2371_ {
                                    } else {
                                        phi_38291_ = true;
                                        break;
                                    }
                                    tile_checkpoints.member[_e2329_] = vec4<f32>(bitcast<f32>(select(_e2338_, 0u, _e2343_)), bitcast<f32>(select(_e2346_, 0u, _e2351_)), bitcast<f32>(select(_e2354_, 0u, _e2359_)), bitcast<f32>(select(_e2362_, 0u, _e2367_)));
                                }
                            }
                        }
                        workgroupBarrier();
                        phi_36006_ = _e1804_;
                        phi_5466_ = type_15_(0u, 2u);
                        loop {
                            let _e2375_ = phi_36006_;
                            let _e2377_ = phi_5466_;
                            if (_e2377_.member < _e2377_.member_1_) {
                                phi_5467_ = type_15_((_e2377_.member + 1u), _e2377_.member_1_);
                                phi_5490_ = type_15_(1u, _e2377_.member);
                            } else {
                                phi_5467_ = _e2377_;
                                phi_5490_ = type_15_(0u, u32());
                            }
                            let _e2390_ = phi_5467_;
                            let _e2392_ = phi_5490_;
                            let _e2396_ = (bitcast<i32>(_e2392_.member) != 0i);
                            if _e2396_ {
                                let _e2398_ = ((_e1821_.member_1_ * 256u) + (_e2392_.member_1_ * 128u));
                                if (_e243_ > _e2398_) {
                                    phi_5505_ = (_e243_ - _e2398_);
                                } else {
                                    phi_5505_ = 0u;
                                }
                                let _e2402_ = phi_5505_;
                                let _e2404_ = select(_e2402_, 128u, (128u < _e2402_));
                                let _e2405_ = (_e2404_ > 0u);
                                if _e2405_ {
                                    let _e2406_ = (_e197_ < _e2404_);
                                    let _e2409_ = ((_e232_ * _e243_) + (_e2398_ + _e197_));
                                    let _e2410_ = (_e2409_ * 2u);
                                    if _e2406_ {
                                        let _e2411_ = (_e2410_ < _e189_);
                                        if _e2411_ {
                                            if _e2411_ {
                                            } else {
                                                phi_38285_ = true;
                                                break;
                                            }
                                            let _e2414_ = core_inputs.member[_e2410_];
                                            phi_5534_ = _e2414_;
                                        } else {
                                            phi_5534_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e2416_ = phi_5534_;
                                        let _e2417_ = (_e2410_ + 1u);
                                        let _e2418_ = (_e2417_ < _e189_);
                                        if _e2418_ {
                                            if _e2418_ {
                                            } else {
                                                phi_38285_ = true;
                                                break;
                                            }
                                            let _e2421_ = core_inputs.member[_e2417_];
                                            phi_5547_ = _e2421_;
                                        } else {
                                            phi_5547_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        }
                                        let _e2423_ = phi_5547_;
                                        phi_5548_ = _e2423_;
                                        phi_5549_ = _e2416_;
                                    } else {
                                        phi_5548_ = vec4<f32>(0f, 0f, 0f, 0f);
                                        phi_5549_ = vec4<f32>(0f, 0f, 0f, 0f);
                                    }
                                    let _e2425_ = phi_5548_;
                                    let _e2427_ = phi_5549_;
                                    phi_36001_ = _e2375_;
                                    phi_5552_ = type_15_(0u, 4u);
                                    loop {
                                        let _e2429_ = phi_36001_;
                                        let _e2431_ = phi_5552_;
                                        if (_e2431_.member < _e2431_.member_1_) {
                                            phi_5553_ = type_15_((_e2431_.member + 1u), _e2431_.member_1_);
                                            phi_5576_ = type_15_(1u, _e2431_.member);
                                        } else {
                                            phi_5553_ = _e2431_;
                                            phi_5576_ = type_15_(0u, u32());
                                        }
                                        let _e2444_ = phi_5553_;
                                        let _e2446_ = phi_5576_;
                                        let _e2450_ = (bitcast<i32>(_e2446_.member) != 0i);
                                        if _e2450_ {
                                            let _e2451_ = (_e2446_.member_1_ & 3u);
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5594_ = _e360_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5594_ = _e360_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5594_ = _e360_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5594_ = _e360_.w;
                                                    break;
                                                }
                                            }
                                            let _e2458_ = phi_5594_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5605_ = _e437_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5605_ = _e437_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5605_ = _e437_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5605_ = _e437_.w;
                                                    break;
                                                }
                                            }
                                            let _e2465_ = phi_5605_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5781_ = _e339_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5781_ = _e339_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5781_ = _e339_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5781_ = _e339_.w;
                                                    break;
                                                }
                                            }
                                            let _e2472_ = phi_5781_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5792_ = _e346_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5792_ = _e346_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5792_ = _e346_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5792_ = _e346_.w;
                                                    break;
                                                }
                                            }
                                            let _e2479_ = phi_5792_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5803_ = _e353_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5803_ = _e353_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5803_ = _e353_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5803_ = _e353_.w;
                                                    break;
                                                }
                                            }
                                            let _e2486_ = phi_5803_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5815_ = _e367_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5815_ = _e367_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5815_ = _e367_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5815_ = _e367_.w;
                                                    break;
                                                }
                                            }
                                            let _e2493_ = phi_5815_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5826_ = _e374_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5826_ = _e374_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5826_ = _e374_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5826_ = _e374_.w;
                                                    break;
                                                }
                                            }
                                            let _e2500_ = phi_5826_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5837_ = _e381_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5837_ = _e381_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5837_ = _e381_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5837_ = _e381_.w;
                                                    break;
                                                }
                                            }
                                            let _e2507_ = phi_5837_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_5848_ = _e388_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_5848_ = _e388_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_5848_ = _e388_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_5848_ = _e388_.w;
                                                    break;
                                                }
                                            }
                                            let _e2514_ = phi_5848_;
                                            switch bitcast<i32>(_e2451_) {
                                                case 0: {
                                                    phi_6063_ = _e444_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_6063_ = _e444_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_6063_ = _e444_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_6063_ = _e444_.w;
                                                    break;
                                                }
                                            }
                                            let _e2521_ = phi_6063_;
                                            if _e2406_ {
                                                let _e2523_ = select(_e2427_, _e2425_, vec4(_e200_));
                                                switch bitcast<i32>(_e2451_) {
                                                    case 0: {
                                                        phi_6581_ = _e2523_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_6581_ = _e2523_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_6581_ = _e2523_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_6581_ = _e2523_.w;
                                                        break;
                                                    }
                                                }
                                                let _e2530_ = phi_6581_;
                                                let _e2531_ = bitcast<u32>(_e2530_);
                                                let _e2532_ = (_e2531_ & 2139095040u);
                                                if (_e2532_ == 2139095040u) {
                                                    phi_6592_ = true;
                                                } else {
                                                    phi_6592_ = (_e2532_ == 0u);
                                                }
                                                let _e2536_ = phi_6592_;
                                                phi_6598_ = bitcast<f32>(select(_e2531_, 0u, _e2536_));
                                            } else {
                                                phi_6598_ = 0f;
                                            }
                                            let _e2540_ = phi_6598_;
                                            if _e200_ {
                                                phi_6603_ = (8u + _e2446_.member_1_);
                                            } else {
                                                phi_6603_ = _e2446_.member_1_;
                                            }
                                            let _e2543_ = phi_6603_;
                                            if _e200_ {
                                                phi_6609_ = (12u + _e2446_.member_1_);
                                            } else {
                                                phi_6609_ = (4u + _e2446_.member_1_);
                                            }
                                            let _e2547_ = phi_6609_;
                                            if _e240_ {
                                                if _e200_ {
                                                    phi_6618_ = (40u + _e2446_.member_1_);
                                                } else {
                                                    phi_6618_ = (32u + _e2446_.member_1_);
                                                }
                                                let _e2767_ = phi_6618_;
                                                if _e200_ {
                                                    phi_6624_ = (44u + _e2446_.member_1_);
                                                } else {
                                                    phi_6624_ = (36u + _e2446_.member_1_);
                                                }
                                                let _e2771_ = phi_6624_;
                                                let _e2773_ = select(0.00001f, _e2465_, (_e2465_ > 0.00001f));
                                                let _e2775_ = select(0f, _e2458_, (_e2458_ > 0f));
                                                if (_e2773_ != _e2773_) {
                                                    phi_24934_ = true;
                                                } else {
                                                    phi_24934_ = (0.00001f >= _e2773_);
                                                }
                                                let _e2779_ = phi_24934_;
                                                let _e2780_ = select(_e2773_, 0.00001f, _e2779_);
                                                if (_e2775_ != _e2775_) {
                                                    phi_24994_ = true;
                                                } else {
                                                    phi_24994_ = (0f >= _e2775_);
                                                }
                                                let _e2784_ = phi_24994_;
                                                let _e2788_ = (1f + (_e2780_ * (_e2780_ + select(_e2775_, 0f, _e2784_))));
                                                let _e2791_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2788_)));
                                                let _e2795_ = (_e2791_ + (_e2791_ * (1f - (_e2788_ * _e2791_))));
                                                let _e2799_ = (_e2795_ + (_e2795_ * (1f - (_e2788_ * _e2795_))));
                                                let _e2800_ = (_e2780_ * _e2799_);
                                                let _e2801_ = (_e2780_ * _e2800_);
                                                if (_e2767_ < 256u) {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                let _e2804_ = aux_scratch[_e2767_];
                                                if (_e2771_ < 256u) {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                let _e2807_ = aux_scratch[_e2771_];
                                                let _e2809_ = select(0u, 256u, _e200_);
                                                let _e2810_ = (_e2800_ + _e2800_);
                                                let _e2811_ = (_e2801_ + _e2801_);
                                                let _e2813_ = ((_e2799_ + _e2799_) - 1f);
                                                let _e2814_ = -(_e2810_);
                                                let _e2815_ = (1f - _e2811_);
                                                if _e2406_ {
                                                    phi_6759_ = (_e2810_ * _e2540_);
                                                } else {
                                                    phi_6759_ = 0f;
                                                }
                                                let _e2818_ = phi_6759_;
                                                if _e2406_ {
                                                    phi_6764_ = (_e2811_ * _e2540_);
                                                } else {
                                                    phi_6764_ = 0f;
                                                }
                                                let _e2821_ = phi_6764_;
                                                let _e2822_ = (_e197_ == 0u);
                                                if _e2822_ {
                                                    if _e2405_ {
                                                        phi_6781_ = (_e2821_ + ((_e2810_ * _e2804_) + (_e2815_ * _e2807_)));
                                                        phi_6782_ = (_e2818_ + ((_e2813_ * _e2804_) + (_e2814_ * _e2807_)));
                                                    } else {
                                                        phi_6781_ = _e2821_;
                                                        phi_6782_ = _e2818_;
                                                    }
                                                    let _e2832_ = phi_6781_;
                                                    let _e2834_ = phi_6782_;
                                                    phi_6783_ = _e2832_;
                                                    phi_6784_ = _e2834_;
                                                } else {
                                                    phi_6783_ = _e2821_;
                                                    phi_6784_ = _e2818_;
                                                }
                                                let _e2836_ = phi_6783_;
                                                let _e2838_ = phi_6784_;
                                                let _e2839_ = (_e2809_ + _e197_);
                                                let _e2840_ = (_e2839_ < 1536u);
                                                if _e2840_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2839_] = _e2838_;
                                                let _e2842_ = (_e2809_ + 128u);
                                                let _e2843_ = (_e2842_ + _e197_);
                                                let _e2844_ = (_e2843_ < 1536u);
                                                if _e2844_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2843_] = _e2836_;
                                                workgroupBarrier();
                                                phi_6801_ = _e2815_;
                                                phi_6804_ = _e2810_;
                                                phi_6806_ = _e2814_;
                                                phi_6808_ = _e2813_;
                                                phi_6810_ = 1u;
                                                phi_6812_ = _e2836_;
                                                phi_6814_ = _e2838_;
                                                loop {
                                                    let _e2847_ = phi_6801_;
                                                    let _e2849_ = phi_6804_;
                                                    let _e2851_ = phi_6806_;
                                                    let _e2853_ = phi_6808_;
                                                    let _e2855_ = phi_6810_;
                                                    let _e2857_ = phi_6812_;
                                                    let _e2859_ = phi_6814_;
                                                    let _e2860_ = (_e2855_ < 128u);
                                                    if _e2860_ {
                                                        if (_e197_ >= _e2855_) {
                                                            let _e2863_ = ((_e197_ - _e2855_) & 127u);
                                                            let _e2864_ = (_e2809_ + _e2863_);
                                                            if (_e2864_ < 1536u) {
                                                            } else {
                                                                phi_36132_ = true;
                                                                break;
                                                            }
                                                            let _e2867_ = scan_scratch[_e2864_];
                                                            let _e2868_ = (_e2842_ + _e2863_);
                                                            if (_e2868_ < 1536u) {
                                                            } else {
                                                                phi_36132_ = true;
                                                                break;
                                                            }
                                                            let _e2871_ = scan_scratch[_e2868_];
                                                            phi_6853_ = (((_e2849_ * _e2867_) + (_e2847_ * _e2871_)) + _e2857_);
                                                            phi_6854_ = (((_e2853_ * _e2867_) + (_e2851_ * _e2871_)) + _e2859_);
                                                        } else {
                                                            phi_6853_ = _e2857_;
                                                            phi_6854_ = _e2859_;
                                                        }
                                                        let _e2881_ = phi_6853_;
                                                        let _e2883_ = phi_6854_;
                                                        workgroupBarrier();
                                                        if _e2840_ {
                                                        } else {
                                                            phi_36132_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2839_] = _e2883_;
                                                        if _e2844_ {
                                                        } else {
                                                            phi_36132_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2843_] = _e2881_;
                                                        let _e2887_ = (_e2853_ + _e2847_);
                                                        workgroupBarrier();
                                                        phi_6802_ = ((_e2849_ * _e2851_) + (_e2847_ * _e2847_));
                                                        phi_6805_ = (_e2849_ * _e2887_);
                                                        phi_6807_ = (_e2851_ * _e2887_);
                                                        phi_6809_ = ((_e2853_ * _e2853_) + (_e2851_ * _e2849_));
                                                        phi_6811_ = (_e2855_ << bitcast<u32>(1i));
                                                        phi_6813_ = _e2881_;
                                                        phi_6815_ = _e2883_;
                                                    } else {
                                                        phi_6802_ = f32();
                                                        phi_6805_ = f32();
                                                        phi_6807_ = f32();
                                                        phi_6809_ = f32();
                                                        phi_6811_ = u32();
                                                        phi_6813_ = f32();
                                                        phi_6815_ = f32();
                                                    }
                                                    let _e2896_ = phi_6802_;
                                                    let _e2898_ = phi_6805_;
                                                    let _e2900_ = phi_6807_;
                                                    let _e2902_ = phi_6809_;
                                                    let _e2904_ = phi_6811_;
                                                    let _e2906_ = phi_6813_;
                                                    let _e2908_ = phi_6815_;
                                                    phi_6801_ = _e2896_;
                                                    phi_6804_ = _e2898_;
                                                    phi_6806_ = _e2900_;
                                                    phi_6808_ = _e2902_;
                                                    phi_6810_ = _e2904_;
                                                    phi_6812_ = _e2906_;
                                                    phi_6814_ = _e2908_;
                                                    phi_36132_ = _e2429_;
                                                    if !(_e2860_) {
                                                        break;
                                                    }
                                                }
                                                let _e2911_ = phi_36132_;
                                                phi_36174_ = _e2911_;
                                                if _e2911_ {
                                                    break;
                                                }
                                                if _e2822_ {
                                                    phi_6906_ = _e2807_;
                                                    phi_6907_ = _e2804_;
                                                } else {
                                                    let _e2913_ = ((_e197_ - 1u) & 127u);
                                                    let _e2914_ = (_e2809_ + _e2913_);
                                                    if (_e2914_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2917_ = scan_scratch[_e2914_];
                                                    let _e2918_ = (_e2842_ + _e2913_);
                                                    if (_e2918_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2921_ = scan_scratch[_e2918_];
                                                    phi_6906_ = _e2921_;
                                                    phi_6907_ = _e2917_;
                                                }
                                                let _e2923_ = phi_6906_;
                                                let _e2925_ = phi_6907_;
                                                if _e2405_ {
                                                    if (_e2404_ <= 128u) {
                                                        phi_6917_ = (_e2404_ - 1u);
                                                    } else {
                                                        phi_6917_ = 127u;
                                                    }
                                                    let _e2929_ = phi_6917_;
                                                    let _e2930_ = (_e2809_ + _e2929_);
                                                    if (_e2930_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2933_ = scan_scratch[_e2930_];
                                                    let _e2934_ = (_e2842_ + _e2929_);
                                                    if (_e2934_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2937_ = scan_scratch[_e2934_];
                                                    phi_6936_ = array<f32, 2>(_e2933_, _e2937_);
                                                } else {
                                                    phi_6936_ = array<f32, 2>(_e2804_, _e2807_);
                                                }
                                                let _e2940_ = phi_6936_;
                                                workgroupBarrier();
                                                let _e2944_ = ((_e2799_ * _e2925_) + (_e2800_ * (_e2540_ - _e2923_)));
                                                let _e2947_ = (_e2944_ * _e2944_);
                                                let _e2951_ = (_e2775_ + (select(0f, _e2521_, (_e2521_ > 0f)) * select(64f, _e2947_, (_e2947_ < 64f))));
                                                let _e2954_ = (1f + (_e2773_ * (_e2773_ + _e2951_)));
                                                let _e2957_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2954_)));
                                                let _e2961_ = (_e2957_ + (_e2957_ * (1f - (_e2954_ * _e2957_))));
                                                let _e2965_ = (_e2961_ + (_e2961_ * (1f - (_e2954_ * _e2961_))));
                                                let _e2966_ = (_e2773_ * _e2965_);
                                                let _e2967_ = (_e2773_ * _e2966_);
                                                if _e2406_ {
                                                    let _e2968_ = (_e2966_ + _e2966_);
                                                    let _e2969_ = (_e2967_ + _e2967_);
                                                    phi_6993_ = type_30_(((_e2965_ + _e2965_) - 1f), -(_e2968_), _e2968_, (1f - _e2969_), (_e2968_ * _e2540_), (_e2969_ * _e2540_));
                                                } else {
                                                    phi_6993_ = type_30_(1f, 0f, 0f, 1f, 0f, 0f);
                                                }
                                                let _e2978_ = phi_6993_;
                                                if (_e2543_ < 256u) {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                let _e2981_ = aux_scratch[_e2543_];
                                                if (_e2547_ < 256u) {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                let _e2984_ = aux_scratch[_e2547_];
                                                let _e2985_ = array<f32, 2>(_e2981_, _e2984_);
                                                let _e2986_ = select(0u, 768u, _e200_);
                                                let _e2987_ = (_e197_ & 127u);
                                                let _e2989_ = (_e2986_ + _e2987_);
                                                let _e2990_ = (_e2989_ < 1536u);
                                                if _e2990_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2989_] = _e2978_.member;
                                                let _e2993_ = (_e2986_ + 128u);
                                                let _e2994_ = (_e2993_ + _e2987_);
                                                let _e2995_ = (_e2994_ < 1536u);
                                                if _e2995_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2994_] = _e2978_.member_1_;
                                                let _e2998_ = (_e2986_ + 256u);
                                                let _e2999_ = (_e2998_ + _e2987_);
                                                let _e3000_ = (_e2999_ < 1536u);
                                                if _e3000_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2999_] = _e2978_.member_2_;
                                                let _e3003_ = (_e2986_ + 384u);
                                                let _e3004_ = (_e3003_ + _e2987_);
                                                let _e3005_ = (_e3004_ < 1536u);
                                                if _e3005_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3004_] = _e2978_.member_3_;
                                                let _e3008_ = (_e2986_ + 512u);
                                                let _e3009_ = (_e3008_ + _e2987_);
                                                let _e3010_ = (_e3009_ < 1536u);
                                                if _e3010_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3009_] = _e2978_.member_4_;
                                                let _e3013_ = (_e2986_ + 640u);
                                                let _e3014_ = (_e3013_ + _e2987_);
                                                let _e3015_ = (_e3014_ < 1536u);
                                                if _e3015_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3014_] = _e2978_.member_5_;
                                                workgroupBarrier();
                                                phi_7071_ = 1u;
                                                phi_7074_ = _e2978_;
                                                loop {
                                                    let _e3018_ = phi_7071_;
                                                    let _e3020_ = phi_7074_;
                                                    let _e3021_ = (_e3018_ < 128u);
                                                    if _e3021_ {
                                                        if (_e197_ >= _e3018_) {
                                                            let _e3024_ = ((_e197_ - _e3018_) & 127u);
                                                            let _e3025_ = (_e2986_ + _e3024_);
                                                            if (_e3025_ < 1536u) {
                                                            } else {
                                                                phi_36161_ = true;
                                                                break;
                                                            }
                                                            let _e3028_ = scan_scratch[_e3025_];
                                                            let _e3029_ = (_e2993_ + _e3024_);
                                                            if (_e3029_ < 1536u) {
                                                            } else {
                                                                phi_36161_ = true;
                                                                break;
                                                            }
                                                            let _e3032_ = scan_scratch[_e3029_];
                                                            let _e3033_ = (_e2998_ + _e3024_);
                                                            if (_e3033_ < 1536u) {
                                                            } else {
                                                                phi_36161_ = true;
                                                                break;
                                                            }
                                                            let _e3036_ = scan_scratch[_e3033_];
                                                            let _e3037_ = (_e3003_ + _e3024_);
                                                            if (_e3037_ < 1536u) {
                                                            } else {
                                                                phi_36161_ = true;
                                                                break;
                                                            }
                                                            let _e3040_ = scan_scratch[_e3037_];
                                                            let _e3041_ = (_e3008_ + _e3024_);
                                                            if (_e3041_ < 1536u) {
                                                            } else {
                                                                phi_36161_ = true;
                                                                break;
                                                            }
                                                            let _e3044_ = scan_scratch[_e3041_];
                                                            let _e3045_ = (_e3013_ + _e3024_);
                                                            if (_e3045_ < 1536u) {
                                                            } else {
                                                                phi_36161_ = true;
                                                                break;
                                                            }
                                                            let _e3048_ = scan_scratch[_e3045_];
                                                            phi_7176_ = type_30_(((_e3020_.member * _e3028_) + (_e3020_.member_1_ * _e3036_)), ((_e3020_.member * _e3032_) + (_e3020_.member_1_ * _e3040_)), ((_e3020_.member_2_ * _e3028_) + (_e3020_.member_3_ * _e3036_)), ((_e3020_.member_2_ * _e3032_) + (_e3020_.member_3_ * _e3040_)), (((_e3020_.member * _e3044_) + (_e3020_.member_1_ * _e3048_)) + _e3020_.member_4_), (((_e3020_.member_2_ * _e3044_) + (_e3020_.member_3_ * _e3048_)) + _e3020_.member_5_));
                                                        } else {
                                                            phi_7176_ = _e3020_;
                                                        }
                                                        let _e3077_ = phi_7176_;
                                                        workgroupBarrier();
                                                        if _e2990_ {
                                                        } else {
                                                            phi_36161_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2989_] = _e3077_.member;
                                                        if _e2995_ {
                                                        } else {
                                                            phi_36161_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2994_] = _e3077_.member_1_;
                                                        if _e3000_ {
                                                        } else {
                                                            phi_36161_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2999_] = _e3077_.member_2_;
                                                        if _e3005_ {
                                                        } else {
                                                            phi_36161_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3004_] = _e3077_.member_3_;
                                                        if _e3010_ {
                                                        } else {
                                                            phi_36161_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3009_] = _e3077_.member_4_;
                                                        if _e3015_ {
                                                        } else {
                                                            phi_36161_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3014_] = _e3077_.member_5_;
                                                        workgroupBarrier();
                                                        phi_7072_ = (_e3018_ << bitcast<u32>(1i));
                                                        phi_7075_ = _e3077_;
                                                    } else {
                                                        phi_7072_ = u32();
                                                        phi_7075_ = type_30_();
                                                    }
                                                    let _e3087_ = phi_7072_;
                                                    let _e3089_ = phi_7075_;
                                                    phi_7071_ = _e3087_;
                                                    phi_7074_ = _e3089_;
                                                    phi_36161_ = _e2911_;
                                                    if !(_e3021_) {
                                                        break;
                                                    }
                                                }
                                                let _e3092_ = phi_36161_;
                                                phi_36174_ = _e3092_;
                                                if _e3092_ {
                                                    break;
                                                }
                                                if _e2822_ {
                                                    phi_7300_ = _e2985_;
                                                } else {
                                                    let _e3094_ = ((_e197_ - 1u) & 127u);
                                                    let _e3095_ = (_e2986_ + _e3094_);
                                                    if (_e3095_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3098_ = scan_scratch[_e3095_];
                                                    let _e3099_ = (_e2993_ + _e3094_);
                                                    if (_e3099_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3102_ = scan_scratch[_e3099_];
                                                    let _e3103_ = (_e2998_ + _e3094_);
                                                    if (_e3103_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3106_ = scan_scratch[_e3103_];
                                                    let _e3107_ = (_e3003_ + _e3094_);
                                                    if (_e3107_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3110_ = scan_scratch[_e3107_];
                                                    let _e3111_ = (_e3008_ + _e3094_);
                                                    if (_e3111_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3114_ = scan_scratch[_e3111_];
                                                    let _e3115_ = (_e3013_ + _e3094_);
                                                    if (_e3115_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3118_ = scan_scratch[_e3115_];
                                                    phi_7300_ = array<f32, 2>((((_e3098_ * _e2981_) + (_e3102_ * _e2984_)) + _e3114_), (((_e3106_ * _e2981_) + (_e3110_ * _e2984_)) + _e3118_));
                                                }
                                                let _e3129_ = phi_7300_;
                                                if _e2405_ {
                                                    if (_e2404_ <= 128u) {
                                                        phi_7312_ = (_e2404_ - 1u);
                                                    } else {
                                                        phi_7312_ = 127u;
                                                    }
                                                    let _e3135_ = phi_7312_;
                                                    let _e3136_ = (_e3135_ & 127u);
                                                    let _e3137_ = (_e2986_ + _e3136_);
                                                    if (_e3137_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3140_ = scan_scratch[_e3137_];
                                                    let _e3141_ = (_e2993_ + _e3136_);
                                                    if (_e3141_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3144_ = scan_scratch[_e3141_];
                                                    let _e3145_ = (_e2998_ + _e3136_);
                                                    if (_e3145_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3148_ = scan_scratch[_e3145_];
                                                    let _e3149_ = (_e3003_ + _e3136_);
                                                    if (_e3149_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3152_ = scan_scratch[_e3149_];
                                                    let _e3153_ = (_e3008_ + _e3136_);
                                                    if (_e3153_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3156_ = scan_scratch[_e3153_];
                                                    let _e3157_ = (_e3013_ + _e3136_);
                                                    if (_e3157_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e3160_ = scan_scratch[_e3157_];
                                                    phi_7376_ = array<f32, 2>((((_e3140_ * _e2981_) + (_e3144_ * _e2984_)) + _e3156_), (((_e3148_ * _e2981_) + (_e3152_ * _e2984_)) + _e3160_));
                                                } else {
                                                    phi_7376_ = _e2985_;
                                                }
                                                let _e3171_ = phi_7376_;
                                                workgroupBarrier();
                                                if _e2406_ {
                                                    let _e3173_ = (_e2540_ - _e3129_[1]);
                                                    let _e3175_ = ((_e2965_ * _e3129_[0]) + (_e2966_ * _e3173_));
                                                    let _e3187_ = select(0u, 2u, _e200_);
                                                    let _e3188_ = (_e2446_.member_1_ * 4u);
                                                    let _e3191_ = (((_e3188_ + _e3187_) * 128u) + _e2987_);
                                                    if (_e3191_ < 2048u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e3191_] = (((((_e3129_[1] + (_e2966_ * _e3129_[0])) + (_e2967_ * _e3173_)) * _e2493_) + (_e3175_ * (_e2500_ + (_e2507_ * _e2951_)))) + (_e2540_ * _e2514_));
                                                    let _e3197_ = (((_e3188_ + (_e3187_ + 1u)) * 128u) + _e2987_);
                                                    if (_e3197_ < 2048u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e3197_] = _e3175_;
                                                }
                                                if _e1827_ {
                                                    let _e3201_ = bitcast<u32>(_e3171_[0]);
                                                    let _e3202_ = (_e3201_ & 2139095040u);
                                                    if (_e3202_ == 2139095040u) {
                                                        phi_7458_ = true;
                                                    } else {
                                                        phi_7458_ = (_e3202_ == 0u);
                                                    }
                                                    let _e3206_ = phi_7458_;
                                                    if (_e2446_.member_1_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e2446_.member_1_] = bitcast<f32>(select(_e3201_, 0u, _e3206_));
                                                    let _e3212_ = bitcast<u32>(_e3171_[1]);
                                                    let _e3213_ = (_e3212_ & 2139095040u);
                                                    if (_e3213_ == 2139095040u) {
                                                        phi_7481_ = true;
                                                    } else {
                                                        phi_7481_ = (_e3213_ == 0u);
                                                    }
                                                    let _e3217_ = phi_7481_;
                                                    let _e3220_ = (4u + _e2446_.member_1_);
                                                    if (_e3220_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3220_] = bitcast<f32>(select(_e3212_, 0u, _e3217_));
                                                    let _e3224_ = bitcast<u32>(_e2940_[0]);
                                                    let _e3225_ = (_e3224_ & 2139095040u);
                                                    if (_e3225_ == 2139095040u) {
                                                        phi_7505_ = true;
                                                    } else {
                                                        phi_7505_ = (_e3225_ == 0u);
                                                    }
                                                    let _e3229_ = phi_7505_;
                                                    let _e3232_ = (32u + _e2446_.member_1_);
                                                    if (_e3232_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3232_] = bitcast<f32>(select(_e3224_, 0u, _e3229_));
                                                    let _e3236_ = bitcast<u32>(_e2940_[1]);
                                                    let _e3237_ = (_e3236_ & 2139095040u);
                                                    if (_e3237_ == 2139095040u) {
                                                        phi_7529_ = true;
                                                    } else {
                                                        phi_7529_ = (_e3237_ == 0u);
                                                    }
                                                    let _e3241_ = phi_7529_;
                                                    let _e3244_ = (36u + _e2446_.member_1_);
                                                    if (_e3244_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3244_] = bitcast<f32>(select(_e3236_, 0u, _e3241_));
                                                }
                                                if (_e196_ == 128u) {
                                                    let _e3249_ = bitcast<u32>(_e3171_[0]);
                                                    let _e3250_ = (_e3249_ & 2139095040u);
                                                    if (_e3250_ == 2139095040u) {
                                                        phi_7557_ = true;
                                                    } else {
                                                        phi_7557_ = (_e3250_ == 0u);
                                                    }
                                                    let _e3254_ = phi_7557_;
                                                    let _e3257_ = (8u + _e2446_.member_1_);
                                                    if (_e3257_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3257_] = bitcast<f32>(select(_e3249_, 0u, _e3254_));
                                                    let _e3261_ = bitcast<u32>(_e3171_[1]);
                                                    let _e3262_ = (_e3261_ & 2139095040u);
                                                    if (_e3262_ == 2139095040u) {
                                                        phi_7581_ = true;
                                                    } else {
                                                        phi_7581_ = (_e3262_ == 0u);
                                                    }
                                                    let _e3266_ = phi_7581_;
                                                    let _e3269_ = (12u + _e2446_.member_1_);
                                                    if (_e3269_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3269_] = bitcast<f32>(select(_e3261_, 0u, _e3266_));
                                                    let _e3273_ = bitcast<u32>(_e2940_[0]);
                                                    let _e3274_ = (_e3273_ & 2139095040u);
                                                    if (_e3274_ == 2139095040u) {
                                                        phi_7605_ = true;
                                                    } else {
                                                        phi_7605_ = (_e3274_ == 0u);
                                                    }
                                                    let _e3278_ = phi_7605_;
                                                    let _e3281_ = (40u + _e2446_.member_1_);
                                                    if (_e3281_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3281_] = bitcast<f32>(select(_e3273_, 0u, _e3278_));
                                                    let _e3285_ = bitcast<u32>(_e2940_[1]);
                                                    let _e3286_ = (_e3285_ & 2139095040u);
                                                    if (_e3286_ == 2139095040u) {
                                                        phi_7629_ = true;
                                                    } else {
                                                        phi_7629_ = (_e3286_ == 0u);
                                                    }
                                                    let _e3290_ = phi_7629_;
                                                    let _e3293_ = (44u + _e2446_.member_1_);
                                                    if (_e3293_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3293_] = bitcast<f32>(select(_e3285_, 0u, _e3290_));
                                                }
                                                workgroupBarrier();
                                                phi_36176_ = _e3092_;
                                            } else {
                                                if (_e2543_ < 256u) {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                let _e2552_ = aux_scratch[_e2543_];
                                                if (_e2547_ < 256u) {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                let _e2555_ = aux_scratch[_e2547_];
                                                let _e2557_ = select(0u, 256u, _e200_);
                                                let _e2558_ = (_e2479_ + _e2479_);
                                                let _e2559_ = (_e2486_ + _e2486_);
                                                let _e2561_ = ((_e2472_ + _e2472_) - 1f);
                                                let _e2562_ = -(_e2558_);
                                                let _e2563_ = (1f - _e2559_);
                                                if _e2406_ {
                                                    phi_7718_ = (_e2558_ * _e2540_);
                                                } else {
                                                    phi_7718_ = 0f;
                                                }
                                                let _e2566_ = phi_7718_;
                                                if _e2406_ {
                                                    phi_7723_ = (_e2559_ * _e2540_);
                                                } else {
                                                    phi_7723_ = 0f;
                                                }
                                                let _e2569_ = phi_7723_;
                                                let _e2570_ = (_e197_ == 0u);
                                                if _e2570_ {
                                                    if _e2405_ {
                                                        phi_7740_ = (_e2569_ + ((_e2558_ * _e2552_) + (_e2563_ * _e2555_)));
                                                        phi_7741_ = (_e2566_ + ((_e2561_ * _e2552_) + (_e2562_ * _e2555_)));
                                                    } else {
                                                        phi_7740_ = _e2569_;
                                                        phi_7741_ = _e2566_;
                                                    }
                                                    let _e2580_ = phi_7740_;
                                                    let _e2582_ = phi_7741_;
                                                    phi_7742_ = _e2580_;
                                                    phi_7743_ = _e2582_;
                                                } else {
                                                    phi_7742_ = _e2569_;
                                                    phi_7743_ = _e2566_;
                                                }
                                                let _e2584_ = phi_7742_;
                                                let _e2586_ = phi_7743_;
                                                let _e2587_ = (_e2557_ + _e197_);
                                                let _e2588_ = (_e2587_ < 1536u);
                                                if _e2588_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2587_] = _e2586_;
                                                let _e2590_ = (_e2557_ + 128u);
                                                let _e2591_ = (_e2590_ + _e197_);
                                                let _e2592_ = (_e2591_ < 1536u);
                                                if _e2592_ {
                                                } else {
                                                    phi_36174_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e2591_] = _e2584_;
                                                workgroupBarrier();
                                                phi_7760_ = _e2563_;
                                                phi_7763_ = _e2558_;
                                                phi_7765_ = _e2562_;
                                                phi_7767_ = _e2561_;
                                                phi_7769_ = 1u;
                                                phi_7771_ = _e2584_;
                                                phi_7773_ = _e2586_;
                                                loop {
                                                    let _e2595_ = phi_7760_;
                                                    let _e2597_ = phi_7763_;
                                                    let _e2599_ = phi_7765_;
                                                    let _e2601_ = phi_7767_;
                                                    let _e2603_ = phi_7769_;
                                                    let _e2605_ = phi_7771_;
                                                    let _e2607_ = phi_7773_;
                                                    let _e2608_ = (_e2603_ < 128u);
                                                    if _e2608_ {
                                                        if (_e197_ >= _e2603_) {
                                                            let _e2611_ = ((_e197_ - _e2603_) & 127u);
                                                            let _e2612_ = (_e2557_ + _e2611_);
                                                            if (_e2612_ < 1536u) {
                                                            } else {
                                                                phi_35980_ = true;
                                                                break;
                                                            }
                                                            let _e2615_ = scan_scratch[_e2612_];
                                                            let _e2616_ = (_e2590_ + _e2611_);
                                                            if (_e2616_ < 1536u) {
                                                            } else {
                                                                phi_35980_ = true;
                                                                break;
                                                            }
                                                            let _e2619_ = scan_scratch[_e2616_];
                                                            phi_7812_ = (((_e2597_ * _e2615_) + (_e2595_ * _e2619_)) + _e2605_);
                                                            phi_7813_ = (((_e2601_ * _e2615_) + (_e2599_ * _e2619_)) + _e2607_);
                                                        } else {
                                                            phi_7812_ = _e2605_;
                                                            phi_7813_ = _e2607_;
                                                        }
                                                        let _e2629_ = phi_7812_;
                                                        let _e2631_ = phi_7813_;
                                                        workgroupBarrier();
                                                        if _e2588_ {
                                                        } else {
                                                            phi_35980_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2587_] = _e2631_;
                                                        if _e2592_ {
                                                        } else {
                                                            phi_35980_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e2591_] = _e2629_;
                                                        let _e2635_ = (_e2601_ + _e2595_);
                                                        workgroupBarrier();
                                                        phi_7761_ = ((_e2597_ * _e2599_) + (_e2595_ * _e2595_));
                                                        phi_7764_ = (_e2597_ * _e2635_);
                                                        phi_7766_ = (_e2599_ * _e2635_);
                                                        phi_7768_ = ((_e2601_ * _e2601_) + (_e2599_ * _e2597_));
                                                        phi_7770_ = (_e2603_ << bitcast<u32>(1i));
                                                        phi_7772_ = _e2629_;
                                                        phi_7774_ = _e2631_;
                                                    } else {
                                                        phi_7761_ = f32();
                                                        phi_7764_ = f32();
                                                        phi_7766_ = f32();
                                                        phi_7768_ = f32();
                                                        phi_7770_ = u32();
                                                        phi_7772_ = f32();
                                                        phi_7774_ = f32();
                                                    }
                                                    let _e2644_ = phi_7761_;
                                                    let _e2646_ = phi_7764_;
                                                    let _e2648_ = phi_7766_;
                                                    let _e2650_ = phi_7768_;
                                                    let _e2652_ = phi_7770_;
                                                    let _e2654_ = phi_7772_;
                                                    let _e2656_ = phi_7774_;
                                                    phi_7760_ = _e2644_;
                                                    phi_7763_ = _e2646_;
                                                    phi_7765_ = _e2648_;
                                                    phi_7767_ = _e2650_;
                                                    phi_7769_ = _e2652_;
                                                    phi_7771_ = _e2654_;
                                                    phi_7773_ = _e2656_;
                                                    phi_35980_ = _e2429_;
                                                    if !(_e2608_) {
                                                        break;
                                                    }
                                                }
                                                let _e2659_ = phi_35980_;
                                                phi_36174_ = _e2659_;
                                                if _e2659_ {
                                                    break;
                                                }
                                                if _e2570_ {
                                                    phi_7865_ = _e2555_;
                                                    phi_7866_ = _e2552_;
                                                } else {
                                                    let _e2661_ = ((_e197_ - 1u) & 127u);
                                                    let _e2662_ = (_e2557_ + _e2661_);
                                                    if (_e2662_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2665_ = scan_scratch[_e2662_];
                                                    let _e2666_ = (_e2590_ + _e2661_);
                                                    if (_e2666_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2669_ = scan_scratch[_e2666_];
                                                    phi_7865_ = _e2669_;
                                                    phi_7866_ = _e2665_;
                                                }
                                                let _e2671_ = phi_7865_;
                                                let _e2673_ = phi_7866_;
                                                if _e2405_ {
                                                    if (_e2404_ <= 128u) {
                                                        phi_7876_ = (_e2404_ - 1u);
                                                    } else {
                                                        phi_7876_ = 127u;
                                                    }
                                                    let _e2677_ = phi_7876_;
                                                    let _e2678_ = (_e2557_ + _e2677_);
                                                    if (_e2678_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2681_ = scan_scratch[_e2678_];
                                                    let _e2682_ = (_e2590_ + _e2677_);
                                                    if (_e2682_ < 1536u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    let _e2685_ = scan_scratch[_e2682_];
                                                    phi_7895_ = array<f32, 2>(_e2681_, _e2685_);
                                                } else {
                                                    phi_7895_ = array<f32, 2>(_e2552_, _e2555_);
                                                }
                                                let _e2688_ = phi_7895_;
                                                workgroupBarrier();
                                                if _e2406_ {
                                                    let _e2689_ = (_e2540_ - _e2671_);
                                                    let _e2692_ = ((_e2472_ * _e2673_) + (_e2479_ * _e2689_));
                                                    let _e2702_ = select(0u, 2u, _e200_);
                                                    let _e2703_ = (_e2446_.member_1_ * 4u);
                                                    let _e2706_ = (_e197_ & 127u);
                                                    let _e2707_ = (((_e2703_ + _e2702_) * 128u) + _e2706_);
                                                    if (_e2707_ < 2048u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e2707_] = (((((_e2671_ + (_e2479_ * _e2673_)) + (_e2486_ * _e2689_)) * _e2493_) + (_e2692_ * (_e2500_ + (_e2507_ * _e2458_)))) + (_e2540_ * _e2514_));
                                                    let _e2713_ = (((_e2703_ + (_e2702_ + 1u)) * 128u) + _e2706_);
                                                    if (_e2713_ < 2048u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e2713_] = _e2692_;
                                                }
                                                if _e1827_ {
                                                    let _e2717_ = bitcast<u32>(_e2688_[0]);
                                                    let _e2718_ = (_e2717_ & 2139095040u);
                                                    if (_e2718_ == 2139095040u) {
                                                        phi_8139_ = true;
                                                    } else {
                                                        phi_8139_ = (_e2718_ == 0u);
                                                    }
                                                    let _e2722_ = phi_8139_;
                                                    if (_e2446_.member_1_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e2446_.member_1_] = bitcast<f32>(select(_e2717_, 0u, _e2722_));
                                                    let _e2728_ = bitcast<u32>(_e2688_[1]);
                                                    let _e2729_ = (_e2728_ & 2139095040u);
                                                    if (_e2729_ == 2139095040u) {
                                                        phi_8162_ = true;
                                                    } else {
                                                        phi_8162_ = (_e2729_ == 0u);
                                                    }
                                                    let _e2733_ = phi_8162_;
                                                    let _e2736_ = (4u + _e2446_.member_1_);
                                                    if (_e2736_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e2736_] = bitcast<f32>(select(_e2728_, 0u, _e2733_));
                                                }
                                                if (_e196_ == 128u) {
                                                    let _e2741_ = bitcast<u32>(_e2688_[0]);
                                                    let _e2742_ = (_e2741_ & 2139095040u);
                                                    if (_e2742_ == 2139095040u) {
                                                        phi_8190_ = true;
                                                    } else {
                                                        phi_8190_ = (_e2742_ == 0u);
                                                    }
                                                    let _e2746_ = phi_8190_;
                                                    let _e2749_ = (8u + _e2446_.member_1_);
                                                    if (_e2749_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e2749_] = bitcast<f32>(select(_e2741_, 0u, _e2746_));
                                                    let _e2753_ = bitcast<u32>(_e2688_[1]);
                                                    let _e2754_ = (_e2753_ & 2139095040u);
                                                    if (_e2754_ == 2139095040u) {
                                                        phi_8214_ = true;
                                                    } else {
                                                        phi_8214_ = (_e2754_ == 0u);
                                                    }
                                                    let _e2758_ = phi_8214_;
                                                    let _e2761_ = (12u + _e2446_.member_1_);
                                                    if (_e2761_ < 256u) {
                                                    } else {
                                                        phi_36174_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e2761_] = bitcast<f32>(select(_e2753_, 0u, _e2758_));
                                                }
                                                workgroupBarrier();
                                                phi_36176_ = _e2659_;
                                            }
                                            let _e3297_ = phi_36176_;
                                            phi_36175_ = _e3297_;
                                        } else {
                                            phi_36175_ = _e2429_;
                                        }
                                        let _e3299_ = phi_36175_;
                                        phi_36001_ = _e3299_;
                                        phi_5552_ = _e2444_;
                                        phi_36174_ = _e3299_;
                                        if !(_e2450_) {
                                            break;
                                        }
                                    }
                                    let _e3302_ = phi_36174_;
                                    phi_38285_ = _e3302_;
                                    if _e3302_ {
                                        break;
                                    }
                                    phi_36241_ = _e3302_;
                                    phi_8229_ = type_15_(0u, 4u);
                                    loop {
                                        let _e3304_ = phi_36241_;
                                        let _e3306_ = phi_8229_;
                                        if (_e3306_.member < _e3306_.member_1_) {
                                            phi_8230_ = type_15_((_e3306_.member + 1u), _e3306_.member_1_);
                                            phi_8253_ = type_15_(1u, _e3306_.member);
                                        } else {
                                            phi_8230_ = _e3306_;
                                            phi_8253_ = type_15_(0u, u32());
                                        }
                                        let _e3319_ = phi_8230_;
                                        let _e3321_ = phi_8253_;
                                        let _e3325_ = (bitcast<i32>(_e3321_.member) != 0i);
                                        if _e3325_ {
                                            let _e3326_ = (_e3321_.member_1_ & 3u);
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8277_ = _e437_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8277_ = _e437_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8277_ = _e437_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8277_ = _e437_.w;
                                                    break;
                                                }
                                            }
                                            let _e3333_ = phi_8277_;
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8288_ = _e395_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8288_ = _e395_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8288_ = _e395_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8288_ = _e395_.w;
                                                    break;
                                                }
                                            }
                                            let _e3340_ = phi_8288_;
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8299_ = _e402_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8299_ = _e402_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8299_ = _e402_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8299_ = _e402_.w;
                                                    break;
                                                }
                                            }
                                            let _e3347_ = phi_8299_;
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8310_ = _e409_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8310_ = _e409_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8310_ = _e409_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8310_ = _e409_.w;
                                                    break;
                                                }
                                            }
                                            let _e3354_ = phi_8310_;
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8321_ = _e416_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8321_ = _e416_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8321_ = _e416_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8321_ = _e416_.w;
                                                    break;
                                                }
                                            }
                                            let _e3361_ = phi_8321_;
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8332_ = _e423_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8332_ = _e423_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8332_ = _e423_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8332_ = _e423_.w;
                                                    break;
                                                }
                                            }
                                            let _e3368_ = phi_8332_;
                                            let _e3369_ = bitcast<u32>(_e3340_);
                                            let _e3370_ = (_e3369_ & 2139095040u);
                                            if (_e3370_ != 2139095040u) {
                                                if (_e3340_ <= 0f) {
                                                    phi_8516_ = 0f;
                                                } else {
                                                    if (_e3340_ >= 0.999995f) {
                                                        phi_8515_ = 0.999995f;
                                                    } else {
                                                        if (_e3370_ == 2139095040u) {
                                                            phi_8509_ = true;
                                                        } else {
                                                            phi_8509_ = (_e3370_ == 0u);
                                                        }
                                                        let _e3379_ = phi_8509_;
                                                        phi_8515_ = bitcast<f32>(select(_e3369_, 0u, _e3379_));
                                                    }
                                                    let _e3383_ = phi_8515_;
                                                    phi_8516_ = _e3383_;
                                                }
                                                let _e3385_ = phi_8516_;
                                                phi_8523_ = _e3385_;
                                            } else {
                                                phi_8523_ = select(0f, 0.999995f, (_e3369_ == 2139095040u));
                                            }
                                            let _e3387_ = phi_8523_;
                                            let _e3388_ = bitcast<u32>(_e3347_);
                                            let _e3389_ = (_e3388_ & 2139095040u);
                                            if (_e3389_ != 2139095040u) {
                                                if (_e3347_ <= 0f) {
                                                    phi_8555_ = 0f;
                                                } else {
                                                    if (_e3347_ >= 1f) {
                                                        phi_8554_ = 1f;
                                                    } else {
                                                        if (_e3389_ == 2139095040u) {
                                                            phi_8548_ = true;
                                                        } else {
                                                            phi_8548_ = (_e3389_ == 0u);
                                                        }
                                                        let _e3398_ = phi_8548_;
                                                        phi_8554_ = bitcast<f32>(select(_e3388_, 0u, _e3398_));
                                                    }
                                                    let _e3402_ = phi_8554_;
                                                    phi_8555_ = _e3402_;
                                                }
                                                let _e3404_ = phi_8555_;
                                                phi_8562_ = _e3404_;
                                            } else {
                                                phi_8562_ = select(0f, 1f, (_e3388_ == 2139095040u));
                                            }
                                            let _e3406_ = phi_8562_;
                                            let _e3407_ = bitcast<u32>(_e3354_);
                                            let _e3408_ = (_e3407_ & 2139095040u);
                                            if (_e3408_ != 2139095040u) {
                                                if (_e3354_ <= 0f) {
                                                    phi_8594_ = 0f;
                                                } else {
                                                    if (_e3354_ >= 1f) {
                                                        phi_8593_ = 1f;
                                                    } else {
                                                        if (_e3408_ == 2139095040u) {
                                                            phi_8587_ = true;
                                                        } else {
                                                            phi_8587_ = (_e3408_ == 0u);
                                                        }
                                                        let _e3417_ = phi_8587_;
                                                        phi_8593_ = bitcast<f32>(select(_e3407_, 0u, _e3417_));
                                                    }
                                                    let _e3421_ = phi_8593_;
                                                    phi_8594_ = _e3421_;
                                                }
                                                let _e3423_ = phi_8594_;
                                                phi_8601_ = _e3423_;
                                            } else {
                                                phi_8601_ = select(0f, 1f, (_e3407_ == 2139095040u));
                                            }
                                            let _e3425_ = phi_8601_;
                                            let _e3426_ = bitcast<u32>(_e3361_);
                                            let _e3427_ = (_e3426_ & 2139095040u);
                                            if (_e3427_ != 2139095040u) {
                                                if (_e3361_ <= 0f) {
                                                    phi_8633_ = 0f;
                                                } else {
                                                    if (_e3361_ >= 64f) {
                                                        phi_8632_ = 64f;
                                                    } else {
                                                        if (_e3427_ == 2139095040u) {
                                                            phi_8626_ = true;
                                                        } else {
                                                            phi_8626_ = (_e3427_ == 0u);
                                                        }
                                                        let _e3436_ = phi_8626_;
                                                        phi_8632_ = bitcast<f32>(select(_e3426_, 0u, _e3436_));
                                                    }
                                                    let _e3440_ = phi_8632_;
                                                    phi_8633_ = _e3440_;
                                                }
                                                let _e3442_ = phi_8633_;
                                                phi_8640_ = _e3442_;
                                            } else {
                                                phi_8640_ = select(0f, 64f, (_e3426_ == 2139095040u));
                                            }
                                            let _e3444_ = phi_8640_;
                                            let _e3445_ = bitcast<u32>(_e3368_);
                                            let _e3446_ = (_e3445_ & 2139095040u);
                                            if (_e3446_ != 2139095040u) {
                                                if (_e3368_ <= 0f) {
                                                    phi_8672_ = 0f;
                                                } else {
                                                    if (_e3368_ >= 1f) {
                                                        phi_8671_ = 1f;
                                                    } else {
                                                        if (_e3446_ == 2139095040u) {
                                                            phi_8665_ = true;
                                                        } else {
                                                            phi_8665_ = (_e3446_ == 0u);
                                                        }
                                                        let _e3455_ = phi_8665_;
                                                        phi_8671_ = bitcast<f32>(select(_e3445_, 0u, _e3455_));
                                                    }
                                                    let _e3459_ = phi_8671_;
                                                    phi_8672_ = _e3459_;
                                                }
                                                let _e3461_ = phi_8672_;
                                                phi_8679_ = _e3461_;
                                            } else {
                                                phi_8679_ = select(0f, 1f, (_e3445_ == 2139095040u));
                                            }
                                            let _e3463_ = phi_8679_;
                                            switch bitcast<i32>(_e3326_) {
                                                case 0: {
                                                    phi_8781_ = _e451_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_8781_ = _e451_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_8781_ = _e451_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_8781_ = _e451_.w;
                                                    break;
                                                }
                                            }
                                            let _e3471_ = phi_8781_;
                                            if (_e3425_ == 0f) {
                                                phi_9281_ = true;
                                            } else {
                                                phi_9281_ = (_e3463_ >= 1f);
                                            }
                                            let _e3475_ = phi_9281_;
                                            if _e3475_ {
                                                let _e3476_ = select(0u, 2u, _e200_);
                                                let _e3477_ = (_e3476_ + 1u);
                                                if _e2406_ {
                                                    let _e3478_ = (_e3321_.member_1_ * 4u);
                                                    let _e3481_ = (_e197_ & 127u);
                                                    let _e3482_ = (((_e3478_ + _e3476_) * 128u) + _e3481_);
                                                    if (_e3482_ < 2048u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    let _e3485_ = lane_buf[_e3482_];
                                                    let _e3488_ = (((_e3478_ + _e3477_) * 128u) + _e3481_);
                                                    if (_e3488_ < 2048u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    let _e3491_ = lane_buf[_e3488_];
                                                    let _e3493_ = bitcast<u32>(_e3491_);
                                                    let _e3494_ = (_e3493_ & 2139095040u);
                                                    if (_e3494_ == 2139095040u) {
                                                        phi_9328_ = true;
                                                    } else {
                                                        phi_9328_ = (_e3494_ == 0u);
                                                    }
                                                    let _e3498_ = phi_9328_;
                                                    let _e3502_ = bitcast<u32>(((_e3333_ + _e3333_) * bitcast<f32>(select(_e3493_, 0u, _e3498_))));
                                                    let _e3503_ = (_e3502_ & 2139095040u);
                                                    if (_e3503_ == 2139095040u) {
                                                        phi_9345_ = true;
                                                    } else {
                                                        phi_9345_ = (_e3503_ == 0u);
                                                    }
                                                    let _e3507_ = phi_9345_;
                                                    let _e3510_ = abs(bitcast<f32>(select(_e3502_, 0u, _e3507_)));
                                                    if (_e3510_ != _e3510_) {
                                                        phi_25054_ = true;
                                                    } else {
                                                        phi_25054_ = (1000000000000000000f <= _e3510_);
                                                    }
                                                    let _e3514_ = phi_25054_;
                                                    if (_e3444_ <= 0f) {
                                                        phi_9358_ = 0f;
                                                    } else {
                                                        phi_9358_ = (_e3444_ * select(_e3510_, 1000000000000000000f, _e3514_));
                                                    }
                                                    let _e3519_ = phi_9358_;
                                                    if (_e3519_ <= 0f) {
                                                        phi_9375_ = 1f;
                                                    } else {
                                                        let _e3521_ = (1f + _e3519_);
                                                        let _e3524_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3521_)));
                                                        let _e3528_ = (_e3524_ + (_e3524_ * (1f - (_e3521_ * _e3524_))));
                                                        phi_9375_ = (_e3528_ + (_e3528_ * (1f - (_e3521_ * _e3528_))));
                                                    }
                                                    let _e3534_ = phi_9375_;
                                                    let _e3536_ = (1f - (_e3406_ * _e3534_));
                                                    let _e3537_ = bitcast<u32>(_e3536_);
                                                    let _e3538_ = (_e3537_ & 2139095040u);
                                                    let _e3539_ = (_e3538_ == 2139095040u);
                                                    if _e3539_ {
                                                        if ((_e3537_ & 2147483647u) > 2139095040u) {
                                                            phi_9422_ = f32();
                                                            phi_9423_ = true;
                                                        } else {
                                                            let _e3554_ = ((_e3537_ & 2147483648u) == 0u);
                                                            if _e3554_ {
                                                                phi_9420_ = 1f;
                                                            } else {
                                                                phi_9420_ = f32();
                                                            }
                                                            let _e3556_ = phi_9420_;
                                                            phi_9422_ = _e3556_;
                                                            phi_9423_ = select(true, false, _e3554_);
                                                        }
                                                        let _e3559_ = phi_9422_;
                                                        let _e3561_ = phi_9423_;
                                                        phi_9502_ = select(_e3559_, 0f, _e3561_);
                                                    } else {
                                                        if (_e3536_ < 0f) {
                                                            phi_9501_ = 0f;
                                                        } else {
                                                            if (_e3536_ > 1f) {
                                                                phi_9500_ = 1f;
                                                            } else {
                                                                if _e3539_ {
                                                                    phi_9494_ = true;
                                                                } else {
                                                                    phi_9494_ = (_e3538_ == 0u);
                                                                }
                                                                let _e3544_ = phi_9494_;
                                                                phi_9500_ = bitcast<f32>(select(_e3537_, 0u, _e3544_));
                                                            }
                                                            let _e3548_ = phi_9500_;
                                                            phi_9501_ = _e3548_;
                                                        }
                                                        let _e3550_ = phi_9501_;
                                                        phi_9502_ = _e3550_;
                                                    }
                                                    let _e3564_ = phi_9502_;
                                                    let _e3565_ = bitcast<u32>((_e3471_ * _e3485_));
                                                    let _e3566_ = (_e3565_ & 2139095040u);
                                                    if (_e3566_ == 2139095040u) {
                                                        phi_9513_ = true;
                                                    } else {
                                                        phi_9513_ = (_e3566_ == 0u);
                                                    }
                                                    let _e3570_ = phi_9513_;
                                                    let _e3572_ = bitcast<f32>(select(_e3565_, 0u, _e3570_));
                                                    let _e3573_ = bitcast<u32>(_e3572_);
                                                    let _e3574_ = (_e3573_ & 2147483647u);
                                                    let _e3577_ = select(0f, _e3572_, (bitcast<f32>(_e3574_) <= 340282350000000000000000000000000000000f));
                                                    let _e3578_ = (_e3577_ * 499.99997f);
                                                    if (_e3578_ != _e3578_) {
                                                        phi_25069_ = true;
                                                    } else {
                                                        phi_25069_ = (0f >= _e3578_);
                                                    }
                                                    let _e3582_ = phi_25069_;
                                                    let _e3583_ = select(_e3578_, 0f, _e3582_);
                                                    if (_e3583_ != _e3583_) {
                                                        phi_25129_ = true;
                                                    } else {
                                                        phi_25129_ = (1f <= _e3583_);
                                                    }
                                                    let _e3587_ = phi_25129_;
                                                    let _e3588_ = select(_e3583_, 1f, _e3587_);
                                                    let _e3594_ = (_e3577_ - 0.002f);
                                                    if (_e3594_ != _e3594_) {
                                                        phi_25189_ = true;
                                                    } else {
                                                        phi_25189_ = (0f >= _e3594_);
                                                    }
                                                    let _e3598_ = phi_25189_;
                                                    let _e3600_ = ((0.001f * (((_e3588_ * _e3588_) * _e3588_) * (2f - _e3588_))) + select(_e3594_, 0f, _e3598_));
                                                    let _e3601_ = (_e3573_ & 2139095040u);
                                                    let _e3602_ = (_e3601_ == 2139095040u);
                                                    if _e3602_ {
                                                        if (_e3574_ > 2139095040u) {
                                                            phi_9665_ = f32();
                                                            phi_9666_ = true;
                                                        } else {
                                                            let _e3616_ = ((_e3573_ & 2147483648u) == 0u);
                                                            if _e3616_ {
                                                                phi_9663_ = 1f;
                                                            } else {
                                                                phi_9663_ = f32();
                                                            }
                                                            let _e3618_ = phi_9663_;
                                                            phi_9665_ = _e3618_;
                                                            phi_9666_ = select(true, false, _e3616_);
                                                        }
                                                        let _e3621_ = phi_9665_;
                                                        let _e3623_ = phi_9666_;
                                                        if _e3623_ {
                                                            phi_9686_ = -1f;
                                                        } else {
                                                            phi_9686_ = _e3621_;
                                                        }
                                                        let _e3625_ = phi_9686_;
                                                        phi_9745_ = _e3625_;
                                                    } else {
                                                        if (_e3572_ < -1f) {
                                                            phi_9744_ = -1f;
                                                        } else {
                                                            if (_e3572_ > 1f) {
                                                                phi_9743_ = 1f;
                                                            } else {
                                                                if _e3602_ {
                                                                    phi_9737_ = true;
                                                                } else {
                                                                    phi_9737_ = (_e3601_ == 0u);
                                                                }
                                                                let _e3607_ = phi_9737_;
                                                                phi_9743_ = bitcast<f32>(select(_e3573_, 0u, _e3607_));
                                                            }
                                                            let _e3611_ = phi_9743_;
                                                            phi_9744_ = _e3611_;
                                                        }
                                                        let _e3613_ = phi_9744_;
                                                        phi_9745_ = _e3613_;
                                                    }
                                                    let _e3627_ = phi_9745_;
                                                    if (_e3463_ <= 0f) {
                                                        phi_9758_ = _e3600_;
                                                    } else {
                                                        phi_9758_ = (((1f - _e3463_) * _e3600_) + (_e3463_ * (_e3627_ * (1f - (0.33333334f * (_e3627_ * _e3627_))))));
                                                    }
                                                    let _e3638_ = phi_9758_;
                                                    let _e3642_ = bitcast<u32>(((1f - (_e3463_ * _e3564_)) * _e3387_));
                                                    let _e3643_ = (_e3642_ & 2139095040u);
                                                    if (_e3643_ == 2139095040u) {
                                                        phi_9772_ = true;
                                                    } else {
                                                        phi_9772_ = (_e3643_ == 0u);
                                                    }
                                                    let _e3647_ = phi_9772_;
                                                    let _e3651_ = bitcast<u32>((_e3564_ * _e3638_));
                                                    let _e3652_ = (_e3651_ & 2139095040u);
                                                    if (_e3652_ == 2139095040u) {
                                                        phi_9789_ = true;
                                                    } else {
                                                        phi_9789_ = (_e3652_ == 0u);
                                                    }
                                                    let _e3656_ = phi_9789_;
                                                    phi_9797_ = type_29_(bitcast<f32>(select(_e3642_, 0u, _e3647_)), bitcast<f32>(select(_e3651_, 0u, _e3656_)));
                                                } else {
                                                    phi_9797_ = type_29_(1f, 0f);
                                                }
                                                let _e3661_ = phi_9797_;
                                                let _e3664_ = ((16u + select(0u, 4u, _e200_)) + _e3321_.member_1_);
                                                if (_e3664_ < 256u) {
                                                } else {
                                                    phi_36342_ = true;
                                                    break;
                                                }
                                                let _e3667_ = aux_scratch[_e3664_];
                                                let _e3670_ = select(0u, 256u, _e200_);
                                                let _e3671_ = (_e197_ & 127u);
                                                let _e3672_ = (_e3670_ + _e3671_);
                                                let _e3673_ = (_e3672_ < 1536u);
                                                if _e3673_ {
                                                } else {
                                                    phi_36342_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3672_] = _e3661_.member;
                                                let _e3675_ = (_e3670_ + 128u);
                                                let _e3676_ = (_e3675_ + _e3671_);
                                                let _e3677_ = (_e3676_ < 1536u);
                                                if _e3677_ {
                                                } else {
                                                    phi_36342_ = true;
                                                    break;
                                                }
                                                scan_scratch[_e3676_] = _e3661_.member_1_;
                                                workgroupBarrier();
                                                phi_9838_ = 1u;
                                                phi_9841_ = _e3661_;
                                                loop {
                                                    let _e3680_ = phi_9838_;
                                                    let _e3682_ = phi_9841_;
                                                    local_24_ = _e3682_;
                                                    local_25_ = _e3682_;
                                                    let _e3683_ = (_e3680_ < 128u);
                                                    if _e3683_ {
                                                        if (_e197_ >= _e3680_) {
                                                            let _e3686_ = ((_e197_ - _e3680_) & 127u);
                                                            let _e3687_ = (_e3670_ + _e3686_);
                                                            if (_e3687_ < 1536u) {
                                                            } else {
                                                                phi_36215_ = true;
                                                                break;
                                                            }
                                                            let _e3690_ = scan_scratch[_e3687_];
                                                            let _e3691_ = (_e3675_ + _e3686_);
                                                            if (_e3691_ < 1536u) {
                                                            } else {
                                                                phi_36215_ = true;
                                                                break;
                                                            }
                                                            let _e3694_ = scan_scratch[_e3691_];
                                                            phi_9881_ = type_29_((_e3682_.member * _e3690_), ((_e3682_.member * _e3694_) + _e3682_.member_1_));
                                                        } else {
                                                            phi_9881_ = _e3682_;
                                                        }
                                                        let _e3702_ = phi_9881_;
                                                        workgroupBarrier();
                                                        if _e3673_ {
                                                        } else {
                                                            phi_36215_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3672_] = _e3702_.member;
                                                        if _e3677_ {
                                                        } else {
                                                            phi_36215_ = true;
                                                            break;
                                                        }
                                                        scan_scratch[_e3676_] = _e3702_.member_1_;
                                                        workgroupBarrier();
                                                        phi_9839_ = (_e3680_ << bitcast<u32>(1i));
                                                        phi_9842_ = _e3702_;
                                                    } else {
                                                        phi_9839_ = u32();
                                                        phi_9842_ = type_29_();
                                                    }
                                                    let _e3708_ = phi_9839_;
                                                    let _e3710_ = phi_9842_;
                                                    phi_9838_ = _e3708_;
                                                    phi_9841_ = _e3710_;
                                                    phi_36215_ = _e3304_;
                                                    if !(_e3683_) {
                                                        break;
                                                    }
                                                }
                                                let _e3713_ = phi_36215_;
                                                phi_36342_ = _e3713_;
                                                if _e3713_ {
                                                    break;
                                                }
                                                let _e3715_ = local_24_;
                                                let _e3718_ = local_25_;
                                                if _e2405_ {
                                                    if (_e2404_ <= 128u) {
                                                        phi_9917_ = (_e2404_ - 1u);
                                                    } else {
                                                        phi_9917_ = 127u;
                                                    }
                                                    let _e3725_ = phi_9917_;
                                                    let _e3726_ = (_e3725_ & 127u);
                                                    let _e3727_ = (_e3670_ + _e3726_);
                                                    if (_e3727_ < 1536u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    let _e3730_ = scan_scratch[_e3727_];
                                                    let _e3731_ = (_e3675_ + _e3726_);
                                                    if (_e3731_ < 1536u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    let _e3734_ = scan_scratch[_e3731_];
                                                    phi_9938_ = ((_e3730_ * _e3667_) + _e3734_);
                                                } else {
                                                    phi_9938_ = _e3667_;
                                                }
                                                let _e3738_ = phi_9938_;
                                                workgroupBarrier();
                                                if _e2406_ {
                                                    let _e3739_ = bitcast<u32>(((_e3715_.member * _e3667_) + _e3718_.member_1_));
                                                    let _e3740_ = (_e3739_ & 2139095040u);
                                                    if (_e3740_ == 2139095040u) {
                                                        phi_9952_ = true;
                                                    } else {
                                                        phi_9952_ = (_e3740_ == 0u);
                                                    }
                                                    let _e3744_ = phi_9952_;
                                                    let _e3750_ = ((((_e3321_.member_1_ * 4u) + _e3477_) * 128u) + _e3671_);
                                                    if (_e3750_ < 2048u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    lane_buf[_e3750_] = bitcast<f32>(select(_e3739_, 0u, _e3744_));
                                                }
                                                if _e1827_ {
                                                    let _e3753_ = bitcast<u32>(_e3738_);
                                                    let _e3754_ = (_e3753_ & 2139095040u);
                                                    if (_e3754_ == 2139095040u) {
                                                        phi_9983_ = true;
                                                    } else {
                                                        phi_9983_ = (_e3754_ == 0u);
                                                    }
                                                    let _e3758_ = phi_9983_;
                                                    let _e3761_ = (16u + _e3321_.member_1_);
                                                    if (_e3761_ < 256u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3761_] = bitcast<f32>(select(_e3753_, 0u, _e3758_));
                                                }
                                                if (_e196_ == 128u) {
                                                    let _e3765_ = bitcast<u32>(_e3738_);
                                                    let _e3766_ = (_e3765_ & 2139095040u);
                                                    if (_e3766_ == 2139095040u) {
                                                        phi_10010_ = true;
                                                    } else {
                                                        phi_10010_ = (_e3766_ == 0u);
                                                    }
                                                    let _e3770_ = phi_10010_;
                                                    let _e3773_ = (20u + _e3321_.member_1_);
                                                    if (_e3773_ < 256u) {
                                                    } else {
                                                        phi_36342_ = true;
                                                        break;
                                                    }
                                                    aux_scratch[_e3773_] = bitcast<f32>(select(_e3765_, 0u, _e3770_));
                                                }
                                                workgroupBarrier();
                                                phi_36344_ = _e3713_;
                                            } else {
                                                phi_36344_ = _e3304_;
                                            }
                                            let _e3777_ = phi_36344_;
                                            phi_36343_ = _e3777_;
                                        } else {
                                            phi_36343_ = _e3304_;
                                        }
                                        let _e3779_ = phi_36343_;
                                        phi_36241_ = _e3779_;
                                        phi_8229_ = _e3319_;
                                        phi_36342_ = _e3779_;
                                        if !(_e3325_) {
                                            break;
                                        }
                                    }
                                    let _e3782_ = phi_36342_;
                                    phi_38285_ = _e3782_;
                                    if _e3782_ {
                                        break;
                                    }
                                    if _e236_ {
                                        phi_36489_ = _e3782_;
                                        phi_10028_ = type_15_(0u, 4u);
                                        loop {
                                            let _e4118_ = phi_36489_;
                                            let _e4120_ = phi_10028_;
                                            if (_e4120_.member < _e4120_.member_1_) {
                                                phi_10029_ = type_15_((_e4120_.member + 1u), _e4120_.member_1_);
                                                phi_10052_ = type_15_(1u, _e4120_.member);
                                            } else {
                                                phi_10029_ = _e4120_;
                                                phi_10052_ = type_15_(0u, u32());
                                            }
                                            let _e4133_ = phi_10029_;
                                            let _e4135_ = phi_10052_;
                                            let _e4139_ = (bitcast<i32>(_e4135_.member) != 0i);
                                            if _e4139_ {
                                                let _e4140_ = (_e4135_.member_1_ & 3u);
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10076_ = _e437_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10076_ = _e437_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10076_ = _e437_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10076_ = _e437_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4147_ = phi_10076_;
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10087_ = _e395_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10087_ = _e395_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10087_ = _e395_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10087_ = _e395_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4154_ = phi_10087_;
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10098_ = _e402_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10098_ = _e402_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10098_ = _e402_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10098_ = _e402_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4161_ = phi_10098_;
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10109_ = _e409_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10109_ = _e409_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10109_ = _e409_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10109_ = _e409_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4168_ = phi_10109_;
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10120_ = _e416_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10120_ = _e416_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10120_ = _e416_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10120_ = _e416_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4175_ = phi_10120_;
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10131_ = _e423_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10131_ = _e423_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10131_ = _e423_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10131_ = _e423_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4182_ = phi_10131_;
                                                let _e4183_ = bitcast<u32>(_e4154_);
                                                let _e4184_ = (_e4183_ & 2139095040u);
                                                if (_e4184_ != 2139095040u) {
                                                    if (_e4154_ <= 0f) {
                                                        phi_10315_ = 0f;
                                                    } else {
                                                        if (_e4154_ >= 0.999995f) {
                                                            phi_10314_ = 0.999995f;
                                                        } else {
                                                            if (_e4184_ == 2139095040u) {
                                                                phi_10308_ = true;
                                                            } else {
                                                                phi_10308_ = (_e4184_ == 0u);
                                                            }
                                                            let _e4193_ = phi_10308_;
                                                            phi_10314_ = bitcast<f32>(select(_e4183_, 0u, _e4193_));
                                                        }
                                                        let _e4197_ = phi_10314_;
                                                        phi_10315_ = _e4197_;
                                                    }
                                                    let _e4199_ = phi_10315_;
                                                    phi_10322_ = _e4199_;
                                                } else {
                                                    phi_10322_ = select(0f, 0.999995f, (_e4183_ == 2139095040u));
                                                }
                                                let _e4201_ = phi_10322_;
                                                let _e4202_ = bitcast<u32>(_e4161_);
                                                let _e4203_ = (_e4202_ & 2139095040u);
                                                if (_e4203_ != 2139095040u) {
                                                    if (_e4161_ <= 0f) {
                                                        phi_10354_ = 0f;
                                                    } else {
                                                        if (_e4161_ >= 1f) {
                                                            phi_10353_ = 1f;
                                                        } else {
                                                            if (_e4203_ == 2139095040u) {
                                                                phi_10347_ = true;
                                                            } else {
                                                                phi_10347_ = (_e4203_ == 0u);
                                                            }
                                                            let _e4212_ = phi_10347_;
                                                            phi_10353_ = bitcast<f32>(select(_e4202_, 0u, _e4212_));
                                                        }
                                                        let _e4216_ = phi_10353_;
                                                        phi_10354_ = _e4216_;
                                                    }
                                                    let _e4218_ = phi_10354_;
                                                    phi_10361_ = _e4218_;
                                                } else {
                                                    phi_10361_ = select(0f, 1f, (_e4202_ == 2139095040u));
                                                }
                                                let _e4220_ = phi_10361_;
                                                let _e4221_ = bitcast<u32>(_e4168_);
                                                let _e4222_ = (_e4221_ & 2139095040u);
                                                if (_e4222_ != 2139095040u) {
                                                    if (_e4168_ <= 0f) {
                                                        phi_10393_ = 0f;
                                                    } else {
                                                        if (_e4168_ >= 1f) {
                                                            phi_10392_ = 1f;
                                                        } else {
                                                            if (_e4222_ == 2139095040u) {
                                                                phi_10386_ = true;
                                                            } else {
                                                                phi_10386_ = (_e4222_ == 0u);
                                                            }
                                                            let _e4231_ = phi_10386_;
                                                            phi_10392_ = bitcast<f32>(select(_e4221_, 0u, _e4231_));
                                                        }
                                                        let _e4235_ = phi_10392_;
                                                        phi_10393_ = _e4235_;
                                                    }
                                                    let _e4237_ = phi_10393_;
                                                    phi_10400_ = _e4237_;
                                                } else {
                                                    phi_10400_ = select(0f, 1f, (_e4221_ == 2139095040u));
                                                }
                                                let _e4239_ = phi_10400_;
                                                let _e4240_ = bitcast<u32>(_e4175_);
                                                let _e4241_ = (_e4240_ & 2139095040u);
                                                if (_e4241_ != 2139095040u) {
                                                    if (_e4175_ <= 0f) {
                                                        phi_10432_ = 0f;
                                                    } else {
                                                        if (_e4175_ >= 64f) {
                                                            phi_10431_ = 64f;
                                                        } else {
                                                            if (_e4241_ == 2139095040u) {
                                                                phi_10425_ = true;
                                                            } else {
                                                                phi_10425_ = (_e4241_ == 0u);
                                                            }
                                                            let _e4250_ = phi_10425_;
                                                            phi_10431_ = bitcast<f32>(select(_e4240_, 0u, _e4250_));
                                                        }
                                                        let _e4254_ = phi_10431_;
                                                        phi_10432_ = _e4254_;
                                                    }
                                                    let _e4256_ = phi_10432_;
                                                    phi_10439_ = _e4256_;
                                                } else {
                                                    phi_10439_ = select(0f, 64f, (_e4240_ == 2139095040u));
                                                }
                                                let _e4258_ = phi_10439_;
                                                let _e4259_ = bitcast<u32>(_e4182_);
                                                let _e4260_ = (_e4259_ & 2139095040u);
                                                if (_e4260_ != 2139095040u) {
                                                    if (_e4182_ <= 0f) {
                                                        phi_10471_ = 0f;
                                                    } else {
                                                        if (_e4182_ >= 1f) {
                                                            phi_10470_ = 1f;
                                                        } else {
                                                            if (_e4260_ == 2139095040u) {
                                                                phi_10464_ = true;
                                                            } else {
                                                                phi_10464_ = (_e4260_ == 0u);
                                                            }
                                                            let _e4269_ = phi_10464_;
                                                            phi_10470_ = bitcast<f32>(select(_e4259_, 0u, _e4269_));
                                                        }
                                                        let _e4273_ = phi_10470_;
                                                        phi_10471_ = _e4273_;
                                                    }
                                                    let _e4275_ = phi_10471_;
                                                    phi_10478_ = _e4275_;
                                                } else {
                                                    phi_10478_ = select(0f, 1f, (_e4259_ == 2139095040u));
                                                }
                                                let _e4277_ = phi_10478_;
                                                let _e4278_ = (_e4147_ + _e4147_);
                                                switch bitcast<i32>(_e4140_) {
                                                    case 0: {
                                                        phi_10595_ = _e451_.x;
                                                        break;
                                                    }
                                                    case 1: {
                                                        phi_10595_ = _e451_.y;
                                                        break;
                                                    }
                                                    case 2: {
                                                        phi_10595_ = _e451_.z;
                                                        break;
                                                    }
                                                    default: {
                                                        phi_10595_ = _e451_.w;
                                                        break;
                                                    }
                                                }
                                                let _e4285_ = phi_10595_;
                                                if (_e4239_ == 0f) {
                                                    phi_11095_ = true;
                                                } else {
                                                    phi_11095_ = (_e4277_ >= 1f);
                                                }
                                                let _e4289_ = phi_11095_;
                                                if _e4289_ {
                                                    phi_36982_ = _e4118_;
                                                } else {
                                                    if _e2406_ {
                                                        let _e4295_ = ((((_e4135_.member_1_ * 4u) + select(0u, 2u, _e200_)) * 128u) + (_e197_ & 127u));
                                                        if (_e4295_ < 2048u) {
                                                        } else {
                                                            phi_36980_ = true;
                                                            break;
                                                        }
                                                        let _e4298_ = lane_buf[_e4295_];
                                                        phi_11122_ = _e4298_;
                                                    } else {
                                                        phi_11122_ = 0f;
                                                    }
                                                    let _e4300_ = phi_11122_;
                                                    if _e2406_ {
                                                        let _e4306_ = ((((_e4135_.member_1_ * 4u) + select(1u, 3u, _e200_)) * 128u) + (_e197_ & 127u));
                                                        if (_e4306_ < 2048u) {
                                                        } else {
                                                            phi_36980_ = true;
                                                            break;
                                                        }
                                                        let _e4309_ = lane_buf[_e4306_];
                                                        phi_11142_ = _e4309_;
                                                    } else {
                                                        phi_11142_ = 0f;
                                                    }
                                                    let _e4311_ = phi_11142_;
                                                    let _e4314_ = ((16u + select(0u, 4u, _e200_)) + _e4135_.member_1_);
                                                    if (_e4314_ < 256u) {
                                                    } else {
                                                        phi_36980_ = true;
                                                        break;
                                                    }
                                                    let _e4317_ = aux_scratch[_e4314_];
                                                    let _e4320_ = (select(512u, 640u, _e200_) + _e197_);
                                                    let _e4321_ = (_e4320_ < 1536u);
                                                    if _e4321_ {
                                                    } else {
                                                        phi_36980_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e4320_] = _e4317_;
                                                    workgroupBarrier();
                                                    phi_36461_ = _e4118_;
                                                    phi_11173_ = type_15_(0u, 6u);
                                                    phi_11176_ = _e4317_;
                                                    phi_11178_ = false;
                                                    loop {
                                                        let _e4324_ = phi_36461_;
                                                        let _e4326_ = phi_11173_;
                                                        let _e4328_ = phi_11176_;
                                                        let _e4330_ = phi_11178_;
                                                        local_29_ = _e4328_;
                                                        local_30_ = _e4328_;
                                                        local_31_ = _e4328_;
                                                        local_32_ = _e4328_;
                                                        if (_e4326_.member < _e4326_.member_1_) {
                                                            phi_11174_ = type_15_((_e4326_.member + 1u), _e4326_.member_1_);
                                                            phi_11201_ = type_15_(1u, _e4326_.member);
                                                        } else {
                                                            phi_11174_ = _e4326_;
                                                            phi_11201_ = type_15_(0u, u32());
                                                        }
                                                        let _e4343_ = phi_11174_;
                                                        let _e4345_ = phi_11201_;
                                                        let _e4348_ = (bitcast<i32>(_e4345_.member) != 0i);
                                                        if _e4348_ {
                                                            if _e4330_ {
                                                                phi_36774_ = _e4324_;
                                                                phi_14186_ = _e4328_;
                                                                phi_14187_ = _e4330_;
                                                            } else {
                                                                let _e4349_ = (_e197_ == 0u);
                                                                if _e4349_ {
                                                                    phi_11224_ = _e4317_;
                                                                } else {
                                                                    let _e4350_ = (_e4320_ - 1u);
                                                                    if (_e4350_ < 1536u) {
                                                                    } else {
                                                                        phi_36772_ = true;
                                                                        break;
                                                                    }
                                                                    let _e4353_ = scan_scratch[_e4350_];
                                                                    phi_11224_ = _e4353_;
                                                                }
                                                                let _e4355_ = phi_11224_;
                                                                if _e2406_ {
                                                                    let _e4356_ = (_e4285_ * _e4300_);
                                                                    let _e4358_ = bitcast<u32>((_e4278_ * _e4311_));
                                                                    let _e4359_ = (_e4358_ & 2139095040u);
                                                                    if (_e4359_ == 2139095040u) {
                                                                        phi_11252_ = true;
                                                                    } else {
                                                                        phi_11252_ = (_e4359_ == 0u);
                                                                    }
                                                                    let _e4363_ = phi_11252_;
                                                                    let _e4368_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e4358_, 0u, _e4363_))) & 2147483647u));
                                                                    if (_e4368_ != _e4368_) {
                                                                        phi_25729_ = true;
                                                                    } else {
                                                                        phi_25729_ = (1000000000000000000f <= _e4368_);
                                                                    }
                                                                    let _e4372_ = phi_25729_;
                                                                    let _e4375_ = (_e4258_ <= 0f);
                                                                    let _e4376_ = select((_e4258_ * select(_e4368_, 1000000000000000000f, _e4372_)), 0f, _e4375_);
                                                                    let _e4377_ = (1f + _e4376_);
                                                                    let _e4380_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4377_)));
                                                                    let _e4384_ = (_e4380_ + (_e4380_ * (1f - (_e4377_ * _e4380_))));
                                                                    let _e4392_ = (1f - (_e4220_ * select((_e4384_ + (_e4384_ * (1f - (_e4377_ * _e4384_)))), 1f, (_e4376_ <= 0f))));
                                                                    if (_e4392_ != _e4392_) {
                                                                        phi_25789_ = true;
                                                                    } else {
                                                                        phi_25789_ = (0f >= _e4392_);
                                                                    }
                                                                    let _e4396_ = phi_25789_;
                                                                    let _e4397_ = select(_e4392_, 0f, _e4396_);
                                                                    if (_e4397_ != _e4397_) {
                                                                        phi_25849_ = true;
                                                                    } else {
                                                                        phi_25849_ = (1f <= _e4397_);
                                                                    }
                                                                    let _e4401_ = phi_25849_;
                                                                    let _e4402_ = select(_e4397_, 1f, _e4401_);
                                                                    let _e4403_ = (_e4201_ * _e4355_);
                                                                    let _e4404_ = (1f - _e4277_);
                                                                    let _e4405_ = (_e4239_ * _e4404_);
                                                                    let _e4407_ = (_e4356_ - (_e4405_ * _e4403_));
                                                                    if (_e4407_ != _e4407_) {
                                                                        phi_25909_ = true;
                                                                    } else {
                                                                        phi_25909_ = (-1f >= _e4407_);
                                                                    }
                                                                    let _e4411_ = phi_25909_;
                                                                    let _e4412_ = select(_e4407_, -1f, _e4411_);
                                                                    if (_e4412_ != _e4412_) {
                                                                        phi_25969_ = true;
                                                                    } else {
                                                                        phi_25969_ = (1f <= _e4412_);
                                                                    }
                                                                    let _e4416_ = phi_25969_;
                                                                    let _e4417_ = select(_e4412_, 1f, _e4416_);
                                                                    let _e4426_ = select(0f, _e4407_, (bitcast<f32>((bitcast<u32>(_e4407_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                                    let _e4427_ = (_e4426_ * 499.99997f);
                                                                    if (_e4427_ != _e4427_) {
                                                                        phi_26029_ = true;
                                                                    } else {
                                                                        phi_26029_ = (0f >= _e4427_);
                                                                    }
                                                                    let _e4431_ = phi_26029_;
                                                                    let _e4432_ = select(_e4427_, 0f, _e4431_);
                                                                    if (_e4432_ != _e4432_) {
                                                                        phi_26089_ = true;
                                                                    } else {
                                                                        phi_26089_ = (1f <= _e4432_);
                                                                    }
                                                                    let _e4436_ = phi_26089_;
                                                                    let _e4437_ = select(_e4432_, 1f, _e4436_);
                                                                    let _e4443_ = (_e4426_ - 0.002f);
                                                                    if (_e4443_ != _e4443_) {
                                                                        phi_26149_ = true;
                                                                    } else {
                                                                        phi_26149_ = (0f >= _e4443_);
                                                                    }
                                                                    let _e4447_ = phi_26149_;
                                                                    let _e4449_ = ((0.001f * (((_e4437_ * _e4437_) * _e4437_) * (2f - _e4437_))) + select(_e4443_, 0f, _e4447_));
                                                                    let _e4453_ = (_e4277_ <= 0f);
                                                                    let _e4460_ = bitcast<u32>((((1f - (_e4277_ * _e4402_)) * _e4403_) + (_e4402_ * select(((_e4404_ * _e4449_) + (_e4277_ * (_e4417_ * (1f - (0.33333334f * (_e4417_ * _e4417_)))))), _e4449_, _e4453_))));
                                                                    let _e4461_ = (_e4460_ & 2139095040u);
                                                                    if (_e4461_ == 2139095040u) {
                                                                        phi_11701_ = true;
                                                                    } else {
                                                                        phi_11701_ = (_e4461_ == 0u);
                                                                    }
                                                                    let _e4465_ = phi_11701_;
                                                                    let _e4468_ = bitcast<u32>(_e4356_);
                                                                    let _e4469_ = (_e4468_ & 2139095040u);
                                                                    if (_e4469_ == 2139095040u) {
                                                                        phi_11749_ = true;
                                                                    } else {
                                                                        phi_11749_ = (_e4469_ == 0u);
                                                                    }
                                                                    let _e4473_ = phi_11749_;
                                                                    let _e4476_ = bitcast<u32>(_e4311_);
                                                                    let _e4477_ = (_e4476_ & 2139095040u);
                                                                    if (_e4477_ == 2139095040u) {
                                                                        phi_11765_ = true;
                                                                    } else {
                                                                        phi_11765_ = (_e4477_ == 0u);
                                                                    }
                                                                    let _e4481_ = phi_11765_;
                                                                    let _e4484_ = bitcast<u32>(_e4355_);
                                                                    let _e4485_ = (_e4484_ & 2139095040u);
                                                                    if (_e4485_ == 2139095040u) {
                                                                        phi_11781_ = true;
                                                                    } else {
                                                                        phi_11781_ = (_e4485_ == 0u);
                                                                    }
                                                                    let _e4489_ = phi_11781_;
                                                                    let _e4493_ = bitcast<u32>((_e4278_ * bitcast<f32>(select(_e4476_, 0u, _e4481_))));
                                                                    let _e4494_ = (_e4493_ & 2139095040u);
                                                                    if (_e4494_ == 2139095040u) {
                                                                        phi_11798_ = true;
                                                                    } else {
                                                                        phi_11798_ = (_e4494_ == 0u);
                                                                    }
                                                                    let _e4498_ = phi_11798_;
                                                                    let _e4501_ = abs(bitcast<f32>(select(_e4493_, 0u, _e4498_)));
                                                                    if (_e4501_ != _e4501_) {
                                                                        phi_26209_ = true;
                                                                    } else {
                                                                        phi_26209_ = (1000000000000000000f <= _e4501_);
                                                                    }
                                                                    let _e4505_ = phi_26209_;
                                                                    if _e4375_ {
                                                                        phi_11811_ = 0f;
                                                                    } else {
                                                                        phi_11811_ = (_e4258_ * select(_e4501_, 1000000000000000000f, _e4505_));
                                                                    }
                                                                    let _e4509_ = phi_11811_;
                                                                    if (_e4509_ <= 0f) {
                                                                        phi_11828_ = 1f;
                                                                    } else {
                                                                        let _e4511_ = (1f + _e4509_);
                                                                        let _e4514_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4511_)));
                                                                        let _e4518_ = (_e4514_ + (_e4514_ * (1f - (_e4511_ * _e4514_))));
                                                                        phi_11828_ = (_e4518_ + (_e4518_ * (1f - (_e4511_ * _e4518_))));
                                                                    }
                                                                    let _e4524_ = phi_11828_;
                                                                    let _e4526_ = (1f - (_e4220_ * _e4524_));
                                                                    let _e4527_ = bitcast<u32>(_e4526_);
                                                                    let _e4528_ = (_e4527_ & 2139095040u);
                                                                    let _e4529_ = (_e4528_ == 2139095040u);
                                                                    if _e4529_ {
                                                                        if ((_e4527_ & 2147483647u) > 2139095040u) {
                                                                            phi_11875_ = f32();
                                                                            phi_11876_ = true;
                                                                        } else {
                                                                            let _e4544_ = ((_e4527_ & 2147483648u) == 0u);
                                                                            if _e4544_ {
                                                                                phi_11873_ = 1f;
                                                                            } else {
                                                                                phi_11873_ = f32();
                                                                            }
                                                                            let _e4546_ = phi_11873_;
                                                                            phi_11875_ = _e4546_;
                                                                            phi_11876_ = select(true, false, _e4544_);
                                                                        }
                                                                        let _e4549_ = phi_11875_;
                                                                        let _e4551_ = phi_11876_;
                                                                        phi_11955_ = select(_e4549_, 0f, _e4551_);
                                                                    } else {
                                                                        if (_e4526_ < 0f) {
                                                                            phi_11954_ = 0f;
                                                                        } else {
                                                                            if (_e4526_ > 1f) {
                                                                                phi_11953_ = 1f;
                                                                            } else {
                                                                                if _e4529_ {
                                                                                    phi_11947_ = true;
                                                                                } else {
                                                                                    phi_11947_ = (_e4528_ == 0u);
                                                                                }
                                                                                let _e4534_ = phi_11947_;
                                                                                phi_11953_ = bitcast<f32>(select(_e4527_, 0u, _e4534_));
                                                                            }
                                                                            let _e4538_ = phi_11953_;
                                                                            phi_11954_ = _e4538_;
                                                                        }
                                                                        let _e4540_ = phi_11954_;
                                                                        phi_11955_ = _e4540_;
                                                                    }
                                                                    let _e4554_ = phi_11955_;
                                                                    let _e4557_ = (bitcast<f32>(select(_e4468_, 0u, _e4473_)) - (_e4405_ * (_e4201_ * bitcast<f32>(select(_e4484_, 0u, _e4489_)))));
                                                                    let _e4563_ = (select(0f, _e4557_, (bitcast<f32>((bitcast<u32>(_e4557_) & 2147483647u)) <= 340282350000000000000000000000000000000f)) * 499.99997f);
                                                                    if (_e4563_ != _e4563_) {
                                                                        phi_26224_ = true;
                                                                    } else {
                                                                        phi_26224_ = (0f >= _e4563_);
                                                                    }
                                                                    let _e4567_ = phi_26224_;
                                                                    let _e4568_ = select(_e4563_, 0f, _e4567_);
                                                                    if (_e4568_ != _e4568_) {
                                                                        phi_26284_ = true;
                                                                    } else {
                                                                        phi_26284_ = (1f <= _e4568_);
                                                                    }
                                                                    let _e4572_ = phi_26284_;
                                                                    let _e4573_ = select(_e4568_, 1f, _e4572_);
                                                                    let _e4577_ = ((_e4573_ * _e4573_) * (3f - (2f * _e4573_)));
                                                                    if (abs(_e4557_) < 1f) {
                                                                        phi_12159_ = (1f - (_e4557_ * _e4557_));
                                                                    } else {
                                                                        phi_12159_ = 0f;
                                                                    }
                                                                    let _e4583_ = phi_12159_;
                                                                    if _e4453_ {
                                                                        phi_12167_ = _e4577_;
                                                                    } else {
                                                                        phi_12167_ = ((_e4404_ * _e4577_) + (_e4277_ * _e4583_));
                                                                    }
                                                                    let _e4588_ = phi_12167_;
                                                                    let _e4595_ = (_e4201_ * ((1f - (_e4277_ * _e4554_)) - (((_e4554_ * _e4239_) * _e4404_) * _e4588_)));
                                                                    let _e4596_ = bitcast<u32>(_e4595_);
                                                                    let _e4597_ = (_e4596_ & 2139095040u);
                                                                    let _e4598_ = (_e4597_ == 2139095040u);
                                                                    if _e4598_ {
                                                                        if ((_e4596_ & 2147483647u) > 2139095040u) {
                                                                            phi_12224_ = f32();
                                                                            phi_12225_ = true;
                                                                        } else {
                                                                            let _e4613_ = ((_e4596_ & 2147483648u) == 0u);
                                                                            if _e4613_ {
                                                                                phi_12222_ = 0.999995f;
                                                                            } else {
                                                                                phi_12222_ = f32();
                                                                            }
                                                                            let _e4615_ = phi_12222_;
                                                                            phi_12224_ = _e4615_;
                                                                            phi_12225_ = select(true, false, _e4613_);
                                                                        }
                                                                        let _e4618_ = phi_12224_;
                                                                        let _e4620_ = phi_12225_;
                                                                        if _e4620_ {
                                                                            phi_12245_ = -0.999995f;
                                                                        } else {
                                                                            phi_12245_ = _e4618_;
                                                                        }
                                                                        let _e4622_ = phi_12245_;
                                                                        phi_12304_ = _e4622_;
                                                                    } else {
                                                                        if (_e4595_ < -0.999995f) {
                                                                            phi_12303_ = -0.999995f;
                                                                        } else {
                                                                            if (_e4595_ > 0.999995f) {
                                                                                phi_12302_ = 0.999995f;
                                                                            } else {
                                                                                if _e4598_ {
                                                                                    phi_12296_ = true;
                                                                                } else {
                                                                                    phi_12296_ = (_e4597_ == 0u);
                                                                                }
                                                                                let _e4603_ = phi_12296_;
                                                                                phi_12302_ = bitcast<f32>(select(_e4596_, 0u, _e4603_));
                                                                            }
                                                                            let _e4607_ = phi_12302_;
                                                                            phi_12303_ = _e4607_;
                                                                        }
                                                                        let _e4609_ = phi_12303_;
                                                                        phi_12304_ = _e4609_;
                                                                    }
                                                                    let _e4624_ = phi_12304_;
                                                                    phi_12307_ = type_29_(bitcast<f32>(select(_e4460_, 0u, _e4465_)), _e4624_);
                                                                } else {
                                                                    phi_12307_ = type_29_(0f, 1f);
                                                                }
                                                                let _e4627_ = phi_12307_;
                                                                if _e2406_ {
                                                                    if ((bitcast<u32>(_e4627_.member) & 2139095040u) != 2139095040u) {
                                                                        let _e4635_ = ((bitcast<u32>(_e4328_) & 2139095040u) != 2139095040u);
                                                                        if _e4635_ {
                                                                            let _e4636_ = abs(_e4627_.member);
                                                                            let _e4637_ = abs(_e4328_);
                                                                            let _e4639_ = select(_e4637_, _e4636_, (_e4636_ > _e4637_));
                                                                            let _e4641_ = select(1f, _e4639_, (_e4639_ > 1f));
                                                                            if ((bitcast<u32>(_e4641_) & 2139095040u) != 2139095040u) {
                                                                                let _e4647_ = (_e4641_ <= 0f);
                                                                                if _e4647_ {
                                                                                    phi_12431_ = f32();
                                                                                } else {
                                                                                    if (_e4641_ < 0.00001f) {
                                                                                        let _e4679_ = (select(0.000000000000000000000001f, _e4641_, (_e4641_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                        if (_e4679_ != _e4679_) {
                                                                                            phi_26464_ = true;
                                                                                        } else {
                                                                                            phi_26464_ = (0.000001f >= _e4679_);
                                                                                        }
                                                                                        let _e4683_ = phi_26464_;
                                                                                        let _e4684_ = select(_e4679_, 0.000001f, _e4683_);
                                                                                        let _e4687_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4684_)));
                                                                                        let _e4691_ = (_e4687_ + (_e4687_ * (1f - (_e4684_ * _e4687_))));
                                                                                        let _e4695_ = (_e4691_ + (_e4691_ * (1f - (_e4684_ * _e4691_))));
                                                                                        let _e4701_ = bitcast<u32>(((_e4695_ + (_e4695_ * (1f - (_e4684_ * _e4695_)))) * 18446744000000000000f));
                                                                                        let _e4702_ = (_e4701_ & 2139095040u);
                                                                                        if (_e4702_ == 2139095040u) {
                                                                                            phi_12389_ = true;
                                                                                        } else {
                                                                                            phi_12389_ = (_e4702_ == 0u);
                                                                                        }
                                                                                        let _e4706_ = phi_12389_;
                                                                                        phi_12430_ = bitcast<f32>(select(_e4701_, 0u, _e4706_));
                                                                                    } else {
                                                                                        if (_e4641_ != _e4641_) {
                                                                                            phi_26404_ = true;
                                                                                        } else {
                                                                                            phi_26404_ = (0.000001f >= _e4641_);
                                                                                        }
                                                                                        let _e4652_ = phi_26404_;
                                                                                        let _e4653_ = select(_e4641_, 0.000001f, _e4652_);
                                                                                        let _e4656_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4653_)));
                                                                                        let _e4660_ = (_e4656_ + (_e4656_ * (1f - (_e4653_ * _e4656_))));
                                                                                        let _e4664_ = (_e4660_ + (_e4660_ * (1f - (_e4653_ * _e4660_))));
                                                                                        let _e4669_ = bitcast<u32>((_e4664_ + (_e4664_ * (1f - (_e4653_ * _e4664_)))));
                                                                                        let _e4670_ = (_e4669_ & 2139095040u);
                                                                                        if (_e4670_ == 2139095040u) {
                                                                                            phi_12424_ = true;
                                                                                        } else {
                                                                                            phi_12424_ = (_e4670_ == 0u);
                                                                                        }
                                                                                        let _e4674_ = phi_12424_;
                                                                                        phi_12430_ = bitcast<f32>(select(_e4669_, 0u, _e4674_));
                                                                                    }
                                                                                    let _e4710_ = phi_12430_;
                                                                                    phi_12431_ = _e4710_;
                                                                                }
                                                                                let _e4712_ = phi_12431_;
                                                                                phi_12432_ = _e4712_;
                                                                                phi_12433_ = _e4647_;
                                                                            } else {
                                                                                phi_12432_ = f32();
                                                                                phi_12433_ = true;
                                                                            }
                                                                            let _e4714_ = phi_12432_;
                                                                            let _e4716_ = phi_12433_;
                                                                            phi_12439_ = (abs((_e4627_.member - _e4328_)) * select(_e4714_, 0f, _e4716_));
                                                                        } else {
                                                                            phi_12439_ = f32();
                                                                        }
                                                                        let _e4720_ = phi_12439_;
                                                                        phi_12441_ = _e4720_;
                                                                        phi_12442_ = select(true, false, _e4635_);
                                                                    } else {
                                                                        phi_12441_ = f32();
                                                                        phi_12442_ = true;
                                                                    }
                                                                    let _e4723_ = phi_12441_;
                                                                    let _e4725_ = phi_12442_;
                                                                    phi_12447_ = select(_e4723_, 1000000000f, _e4725_);
                                                                } else {
                                                                    phi_12447_ = 0f;
                                                                }
                                                                let _e4728_ = phi_12447_;
                                                                let _e4729_ = (_e196_ < 128u);
                                                                if _e4729_ {
                                                                    let _e4730_ = (48u + _e197_);
                                                                    if (_e4730_ < 256u) {
                                                                    } else {
                                                                        phi_36772_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e4730_] = _e4728_;
                                                                }
                                                                workgroupBarrier();
                                                                let _e4733_ = (_e196_ >= 128u);
                                                                if _e4733_ {
                                                                    let _e4734_ = (48u + _e197_);
                                                                    let _e4735_ = (_e4734_ < 256u);
                                                                    if _e4735_ {
                                                                    } else {
                                                                        phi_36772_ = true;
                                                                        break;
                                                                    }
                                                                    let _e4737_ = aux_scratch[_e4734_];
                                                                    if _e4735_ {
                                                                    } else {
                                                                        phi_36772_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e4734_] = select(_e4737_, _e4728_, (_e4728_ > _e4737_));
                                                                }
                                                                workgroupBarrier();
                                                                phi_12485_ = 64u;
                                                                loop {
                                                                    let _e4741_ = phi_12485_;
                                                                    let _e4742_ = (_e4741_ > 0u);
                                                                    if _e4742_ {
                                                                        if (_e197_ < _e4741_) {
                                                                            if _e4729_ {
                                                                                let _e4744_ = (48u + _e197_);
                                                                                let _e4745_ = (_e4744_ < 256u);
                                                                                if _e4745_ {
                                                                                } else {
                                                                                    phi_36452_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e4747_ = aux_scratch[_e4744_];
                                                                                let _e4748_ = (_e4744_ + _e4741_);
                                                                                if (_e4748_ < 256u) {
                                                                                } else {
                                                                                    phi_36452_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e4751_ = aux_scratch[_e4748_];
                                                                                if _e4745_ {
                                                                                } else {
                                                                                    phi_36452_ = true;
                                                                                    break;
                                                                                }
                                                                                aux_scratch[_e4744_] = select(_e4747_, _e4751_, (_e4751_ > _e4747_));
                                                                            }
                                                                        }
                                                                        workgroupBarrier();
                                                                        phi_12486_ = (_e4741_ >> bitcast<u32>(1i));
                                                                    } else {
                                                                        phi_12486_ = u32();
                                                                    }
                                                                    let _e4757_ = phi_12486_;
                                                                    phi_12485_ = _e4757_;
                                                                    phi_36452_ = _e4324_;
                                                                    if !(_e4742_) {
                                                                        break;
                                                                    }
                                                                }
                                                                let _e4760_ = phi_36452_;
                                                                phi_36772_ = _e4760_;
                                                                if _e4760_ {
                                                                    break;
                                                                }
                                                                let _e4762_ = aux_scratch[48];
                                                                workgroupBarrier();
                                                                if (_e4762_ <= 0.0000005f) {
                                                                    phi_36775_ = _e4760_;
                                                                    phi_14184_ = _e4328_;
                                                                    phi_14185_ = true;
                                                                } else {
                                                                    if _e2406_ {
                                                                        phi_12545_ = type_29_(_e4627_.member_1_, (_e4627_.member - (_e4627_.member_1_ * _e4355_)));
                                                                    } else {
                                                                        phi_12545_ = type_29_(1f, 0f);
                                                                    }
                                                                    let _e4768_ = phi_12545_;
                                                                    let _e4771_ = select(0u, 256u, _e200_);
                                                                    let _e4772_ = (_e197_ & 127u);
                                                                    let _e4773_ = (_e4771_ + _e4772_);
                                                                    let _e4774_ = (_e4773_ < 1536u);
                                                                    if _e4774_ {
                                                                    } else {
                                                                        phi_36772_ = true;
                                                                        break;
                                                                    }
                                                                    scan_scratch[_e4773_] = _e4768_.member;
                                                                    let _e4776_ = (_e4771_ + 128u);
                                                                    let _e4777_ = (_e4776_ + _e4772_);
                                                                    let _e4778_ = (_e4777_ < 1536u);
                                                                    if _e4778_ {
                                                                    } else {
                                                                        phi_36772_ = true;
                                                                        break;
                                                                    }
                                                                    scan_scratch[_e4777_] = _e4768_.member_1_;
                                                                    workgroupBarrier();
                                                                    phi_12573_ = 1u;
                                                                    phi_12576_ = _e4768_;
                                                                    loop {
                                                                        let _e4781_ = phi_12573_;
                                                                        let _e4783_ = phi_12576_;
                                                                        local_27_ = _e4783_;
                                                                        local_28_ = _e4783_;
                                                                        let _e4784_ = (_e4781_ < 128u);
                                                                        if _e4784_ {
                                                                            if (_e197_ >= _e4781_) {
                                                                                let _e4787_ = ((_e197_ - _e4781_) & 127u);
                                                                                let _e4788_ = (_e4771_ + _e4787_);
                                                                                if (_e4788_ < 1536u) {
                                                                                } else {
                                                                                    phi_36661_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e4791_ = scan_scratch[_e4788_];
                                                                                let _e4792_ = (_e4776_ + _e4787_);
                                                                                if (_e4792_ < 1536u) {
                                                                                } else {
                                                                                    phi_36661_ = true;
                                                                                    break;
                                                                                }
                                                                                let _e4795_ = scan_scratch[_e4792_];
                                                                                phi_12616_ = type_29_((_e4783_.member * _e4791_), ((_e4783_.member * _e4795_) + _e4783_.member_1_));
                                                                            } else {
                                                                                phi_12616_ = _e4783_;
                                                                            }
                                                                            let _e4803_ = phi_12616_;
                                                                            workgroupBarrier();
                                                                            if _e4774_ {
                                                                            } else {
                                                                                phi_36661_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e4773_] = _e4803_.member;
                                                                            if _e4778_ {
                                                                            } else {
                                                                                phi_36661_ = true;
                                                                                break;
                                                                            }
                                                                            scan_scratch[_e4777_] = _e4803_.member_1_;
                                                                            workgroupBarrier();
                                                                            phi_12574_ = (_e4781_ << bitcast<u32>(1i));
                                                                            phi_12577_ = _e4803_;
                                                                        } else {
                                                                            phi_12574_ = u32();
                                                                            phi_12577_ = type_29_();
                                                                        }
                                                                        let _e4809_ = phi_12574_;
                                                                        let _e4811_ = phi_12577_;
                                                                        phi_12573_ = _e4809_;
                                                                        phi_12576_ = _e4811_;
                                                                        phi_36661_ = _e4760_;
                                                                        if !(_e4784_) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    let _e4814_ = phi_36661_;
                                                                    phi_36772_ = _e4814_;
                                                                    if _e4814_ {
                                                                        break;
                                                                    }
                                                                    let _e4816_ = local_27_;
                                                                    let _e4819_ = local_28_;
                                                                    if _e2405_ {
                                                                        if (_e2404_ <= 128u) {
                                                                            phi_12652_ = (_e2404_ - 1u);
                                                                        } else {
                                                                            phi_12652_ = 127u;
                                                                        }
                                                                        let _e4826_ = phi_12652_;
                                                                        let _e4827_ = (_e4826_ & 127u);
                                                                        if ((_e4771_ + _e4827_) < 1536u) {
                                                                        } else {
                                                                            phi_36772_ = true;
                                                                            break;
                                                                        }
                                                                        if ((_e4776_ + _e4827_) < 1536u) {
                                                                        } else {
                                                                            phi_36772_ = true;
                                                                            break;
                                                                        }
                                                                    }
                                                                    workgroupBarrier();
                                                                    phi_36676_ = _e4814_;
                                                                    phi_12674_ = type_34_(0i, 3i);
                                                                    phi_12677_ = _e4328_;
                                                                    phi_12679_ = 1f;
                                                                    phi_12681_ = false;
                                                                    phi_12683_ = _e4330_;
                                                                    loop {
                                                                        let _e4834_ = phi_36676_;
                                                                        let _e4836_ = phi_12674_;
                                                                        let _e4838_ = phi_12677_;
                                                                        let _e4840_ = phi_12679_;
                                                                        let _e4842_ = phi_12681_;
                                                                        let _e4844_ = phi_12683_;
                                                                        local_36_ = _e4838_;
                                                                        local_37_ = _e4844_;
                                                                        if (_e4836_.member < _e4836_.member_1_) {
                                                                            let _e4851_ = (_e4836_.member + 1i);
                                                                            if select(false, true, ((false == (_e4851_ > _e4836_.member)) != false)) {
                                                                                phi_12710_ = type_35_(0u, i32());
                                                                            } else {
                                                                                phi_12710_ = type_35_(1u, _e4851_);
                                                                            }
                                                                            let _e4861_ = phi_12710_;
                                                                            if (bitcast<i32>(_e4861_.member) != 0i) {
                                                                            } else {
                                                                                phi_36766_ = true;
                                                                                break;
                                                                            }
                                                                            phi_12675_ = type_34_(_e4861_.member_1_, _e4836_.member_1_);
                                                                            phi_12724_ = type_35_(1u, _e4836_.member);
                                                                        } else {
                                                                            phi_12675_ = _e4836_;
                                                                            phi_12724_ = type_35_(0u, i32());
                                                                        }
                                                                        let _e4871_ = phi_12675_;
                                                                        let _e4873_ = phi_12724_;
                                                                        let _e4877_ = (bitcast<i32>(_e4873_.member) != 0i);
                                                                        if _e4877_ {
                                                                            if _e4842_ {
                                                                                phi_36768_ = _e4834_;
                                                                                phi_14180_ = _e4838_;
                                                                                phi_14181_ = _e4840_;
                                                                                phi_14182_ = _e4842_;
                                                                                phi_14183_ = _e4844_;
                                                                            } else {
                                                                                if _e2406_ {
                                                                                    let _e4879_ = (_e4838_ + (_e4840_ * (((_e4816_.member * _e4317_) + _e4819_.member_1_) - _e4328_)));
                                                                                    let _e4880_ = bitcast<u32>(_e4879_);
                                                                                    let _e4881_ = (_e4880_ & 2139095040u);
                                                                                    let _e4882_ = (_e4881_ == 2139095040u);
                                                                                    if _e4882_ {
                                                                                        if ((_e4880_ & 2147483647u) > 2139095040u) {
                                                                                            phi_12784_ = f32();
                                                                                            phi_12785_ = true;
                                                                                        } else {
                                                                                            let _e4897_ = ((_e4880_ & 2147483648u) == 0u);
                                                                                            if _e4897_ {
                                                                                                phi_12782_ = 1000000f;
                                                                                            } else {
                                                                                                phi_12782_ = f32();
                                                                                            }
                                                                                            let _e4899_ = phi_12782_;
                                                                                            phi_12784_ = _e4899_;
                                                                                            phi_12785_ = select(true, false, _e4897_);
                                                                                        }
                                                                                        let _e4902_ = phi_12784_;
                                                                                        let _e4904_ = phi_12785_;
                                                                                        if _e4904_ {
                                                                                            phi_12805_ = -1000000f;
                                                                                        } else {
                                                                                            phi_12805_ = _e4902_;
                                                                                        }
                                                                                        let _e4906_ = phi_12805_;
                                                                                        phi_12864_ = _e4906_;
                                                                                    } else {
                                                                                        if (_e4879_ < -1000000f) {
                                                                                            phi_12863_ = -1000000f;
                                                                                        } else {
                                                                                            if (_e4879_ > 1000000f) {
                                                                                                phi_12862_ = 1000000f;
                                                                                            } else {
                                                                                                if _e4882_ {
                                                                                                    phi_12856_ = true;
                                                                                                } else {
                                                                                                    phi_12856_ = (_e4881_ == 0u);
                                                                                                }
                                                                                                let _e4887_ = phi_12856_;
                                                                                                phi_12862_ = bitcast<f32>(select(_e4880_, 0u, _e4887_));
                                                                                            }
                                                                                            let _e4891_ = phi_12862_;
                                                                                            phi_12863_ = _e4891_;
                                                                                        }
                                                                                        let _e4893_ = phi_12863_;
                                                                                        phi_12864_ = _e4893_;
                                                                                    }
                                                                                    let _e4908_ = phi_12864_;
                                                                                    phi_12865_ = _e4908_;
                                                                                } else {
                                                                                    phi_12865_ = _e4317_;
                                                                                }
                                                                                let _e4910_ = phi_12865_;
                                                                                let _e4911_ = (select(768u, 896u, _e200_) + _e197_);
                                                                                if (_e4911_ < 1536u) {
                                                                                } else {
                                                                                    phi_36766_ = true;
                                                                                    break;
                                                                                }
                                                                                scan_scratch[_e4911_] = _e4910_;
                                                                                workgroupBarrier();
                                                                                if _e4349_ {
                                                                                    phi_12886_ = _e4317_;
                                                                                } else {
                                                                                    let _e4914_ = (_e4911_ - 1u);
                                                                                    if (_e4914_ < 1536u) {
                                                                                    } else {
                                                                                        phi_36766_ = true;
                                                                                        break;
                                                                                    }
                                                                                    let _e4917_ = scan_scratch[_e4914_];
                                                                                    phi_12886_ = _e4917_;
                                                                                }
                                                                                let _e4919_ = phi_12886_;
                                                                                if _e2406_ {
                                                                                    let _e4922_ = bitcast<u32>((_e4278_ * _e4311_));
                                                                                    let _e4923_ = (_e4922_ & 2139095040u);
                                                                                    if (_e4923_ == 2139095040u) {
                                                                                        phi_12914_ = true;
                                                                                    } else {
                                                                                        phi_12914_ = (_e4923_ == 0u);
                                                                                    }
                                                                                    let _e4927_ = phi_12914_;
                                                                                    let _e4932_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e4922_, 0u, _e4927_))) & 2147483647u));
                                                                                    if (_e4932_ != _e4932_) {
                                                                                        phi_26524_ = true;
                                                                                    } else {
                                                                                        phi_26524_ = (1000000000000000000f <= _e4932_);
                                                                                    }
                                                                                    let _e4936_ = phi_26524_;
                                                                                    let _e4940_ = select((_e4258_ * select(_e4932_, 1000000000000000000f, _e4936_)), 0f, (_e4258_ <= 0f));
                                                                                    let _e4941_ = (1f + _e4940_);
                                                                                    let _e4944_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4941_)));
                                                                                    let _e4948_ = (_e4944_ + (_e4944_ * (1f - (_e4941_ * _e4944_))));
                                                                                    let _e4956_ = (1f - (_e4220_ * select((_e4948_ + (_e4948_ * (1f - (_e4941_ * _e4948_)))), 1f, (_e4940_ <= 0f))));
                                                                                    if (_e4956_ != _e4956_) {
                                                                                        phi_26584_ = true;
                                                                                    } else {
                                                                                        phi_26584_ = (0f >= _e4956_);
                                                                                    }
                                                                                    let _e4960_ = phi_26584_;
                                                                                    let _e4961_ = select(_e4956_, 0f, _e4960_);
                                                                                    if (_e4961_ != _e4961_) {
                                                                                        phi_26644_ = true;
                                                                                    } else {
                                                                                        phi_26644_ = (1f <= _e4961_);
                                                                                    }
                                                                                    let _e4965_ = phi_26644_;
                                                                                    let _e4966_ = select(_e4961_, 1f, _e4965_);
                                                                                    let _e4967_ = (_e4201_ * _e4919_);
                                                                                    let _e4968_ = (1f - _e4277_);
                                                                                    let _e4971_ = ((_e4285_ * _e4300_) - ((_e4239_ * _e4968_) * _e4967_));
                                                                                    if (_e4971_ != _e4971_) {
                                                                                        phi_26704_ = true;
                                                                                    } else {
                                                                                        phi_26704_ = (-1f >= _e4971_);
                                                                                    }
                                                                                    let _e4975_ = phi_26704_;
                                                                                    let _e4976_ = select(_e4971_, -1f, _e4975_);
                                                                                    if (_e4976_ != _e4976_) {
                                                                                        phi_26764_ = true;
                                                                                    } else {
                                                                                        phi_26764_ = (1f <= _e4976_);
                                                                                    }
                                                                                    let _e4980_ = phi_26764_;
                                                                                    let _e4981_ = select(_e4976_, 1f, _e4980_);
                                                                                    let _e4990_ = select(0f, _e4971_, (bitcast<f32>((bitcast<u32>(_e4971_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                                                    let _e4991_ = (_e4990_ * 499.99997f);
                                                                                    if (_e4991_ != _e4991_) {
                                                                                        phi_26824_ = true;
                                                                                    } else {
                                                                                        phi_26824_ = (0f >= _e4991_);
                                                                                    }
                                                                                    let _e4995_ = phi_26824_;
                                                                                    let _e4996_ = select(_e4991_, 0f, _e4995_);
                                                                                    if (_e4996_ != _e4996_) {
                                                                                        phi_26884_ = true;
                                                                                    } else {
                                                                                        phi_26884_ = (1f <= _e4996_);
                                                                                    }
                                                                                    let _e5000_ = phi_26884_;
                                                                                    let _e5001_ = select(_e4996_, 1f, _e5000_);
                                                                                    let _e5007_ = (_e4990_ - 0.002f);
                                                                                    if (_e5007_ != _e5007_) {
                                                                                        phi_26944_ = true;
                                                                                    } else {
                                                                                        phi_26944_ = (0f >= _e5007_);
                                                                                    }
                                                                                    let _e5011_ = phi_26944_;
                                                                                    let _e5013_ = ((0.001f * (((_e5001_ * _e5001_) * _e5001_) * (2f - _e5001_))) + select(_e5007_, 0f, _e5011_));
                                                                                    let _e5024_ = bitcast<u32>((((1f - (_e4277_ * _e4966_)) * _e4967_) + (_e4966_ * select(((_e4968_ * _e5013_) + (_e4277_ * (_e4981_ * (1f - (0.33333334f * (_e4981_ * _e4981_)))))), _e5013_, (_e4277_ <= 0f)))));
                                                                                    let _e5025_ = (_e5024_ & 2139095040u);
                                                                                    if (_e5025_ == 2139095040u) {
                                                                                        phi_13363_ = true;
                                                                                    } else {
                                                                                        phi_13363_ = (_e5025_ == 0u);
                                                                                    }
                                                                                    let _e5029_ = phi_13363_;
                                                                                    let _e5031_ = bitcast<f32>(select(_e5024_, 0u, _e5029_));
                                                                                    if ((bitcast<u32>(_e5031_) & 2139095040u) != 2139095040u) {
                                                                                        let _e5037_ = ((bitcast<u32>(_e4910_) & 2139095040u) != 2139095040u);
                                                                                        if _e5037_ {
                                                                                            let _e5038_ = abs(_e5031_);
                                                                                            let _e5039_ = abs(_e4910_);
                                                                                            let _e5041_ = select(_e5039_, _e5038_, (_e5038_ > _e5039_));
                                                                                            let _e5043_ = select(1f, _e5041_, (_e5041_ > 1f));
                                                                                            if ((bitcast<u32>(_e5043_) & 2139095040u) != 2139095040u) {
                                                                                                let _e5049_ = (_e5043_ <= 0f);
                                                                                                if _e5049_ {
                                                                                                    phi_14049_ = f32();
                                                                                                } else {
                                                                                                    if (_e5043_ < 0.00001f) {
                                                                                                        let _e5081_ = (select(0.000000000000000000000001f, _e5043_, (_e5043_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                                                        if (_e5081_ != _e5081_) {
                                                                                                            phi_27259_ = true;
                                                                                                        } else {
                                                                                                            phi_27259_ = (0.000001f >= _e5081_);
                                                                                                        }
                                                                                                        let _e5085_ = phi_27259_;
                                                                                                        let _e5086_ = select(_e5081_, 0.000001f, _e5085_);
                                                                                                        let _e5089_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5086_)));
                                                                                                        let _e5093_ = (_e5089_ + (_e5089_ * (1f - (_e5086_ * _e5089_))));
                                                                                                        let _e5097_ = (_e5093_ + (_e5093_ * (1f - (_e5086_ * _e5093_))));
                                                                                                        let _e5103_ = bitcast<u32>(((_e5097_ + (_e5097_ * (1f - (_e5086_ * _e5097_)))) * 18446744000000000000f));
                                                                                                        let _e5104_ = (_e5103_ & 2139095040u);
                                                                                                        if (_e5104_ == 2139095040u) {
                                                                                                            phi_14007_ = true;
                                                                                                        } else {
                                                                                                            phi_14007_ = (_e5104_ == 0u);
                                                                                                        }
                                                                                                        let _e5108_ = phi_14007_;
                                                                                                        phi_14048_ = bitcast<f32>(select(_e5103_, 0u, _e5108_));
                                                                                                    } else {
                                                                                                        if (_e5043_ != _e5043_) {
                                                                                                            phi_27199_ = true;
                                                                                                        } else {
                                                                                                            phi_27199_ = (0.000001f >= _e5043_);
                                                                                                        }
                                                                                                        let _e5054_ = phi_27199_;
                                                                                                        let _e5055_ = select(_e5043_, 0.000001f, _e5054_);
                                                                                                        let _e5058_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5055_)));
                                                                                                        let _e5062_ = (_e5058_ + (_e5058_ * (1f - (_e5055_ * _e5058_))));
                                                                                                        let _e5066_ = (_e5062_ + (_e5062_ * (1f - (_e5055_ * _e5062_))));
                                                                                                        let _e5071_ = bitcast<u32>((_e5066_ + (_e5066_ * (1f - (_e5055_ * _e5066_)))));
                                                                                                        let _e5072_ = (_e5071_ & 2139095040u);
                                                                                                        if (_e5072_ == 2139095040u) {
                                                                                                            phi_14042_ = true;
                                                                                                        } else {
                                                                                                            phi_14042_ = (_e5072_ == 0u);
                                                                                                        }
                                                                                                        let _e5076_ = phi_14042_;
                                                                                                        phi_14048_ = bitcast<f32>(select(_e5071_, 0u, _e5076_));
                                                                                                    }
                                                                                                    let _e5112_ = phi_14048_;
                                                                                                    phi_14049_ = _e5112_;
                                                                                                }
                                                                                                let _e5114_ = phi_14049_;
                                                                                                phi_14050_ = _e5114_;
                                                                                                phi_14051_ = _e5049_;
                                                                                            } else {
                                                                                                phi_14050_ = f32();
                                                                                                phi_14051_ = true;
                                                                                            }
                                                                                            let _e5116_ = phi_14050_;
                                                                                            let _e5118_ = phi_14051_;
                                                                                            phi_14057_ = (abs((_e5031_ - _e4910_)) * select(_e5116_, 0f, _e5118_));
                                                                                        } else {
                                                                                            phi_14057_ = f32();
                                                                                        }
                                                                                        let _e5122_ = phi_14057_;
                                                                                        phi_14059_ = _e5122_;
                                                                                        phi_14060_ = select(true, false, _e5037_);
                                                                                    } else {
                                                                                        phi_14059_ = f32();
                                                                                        phi_14060_ = true;
                                                                                    }
                                                                                    let _e5125_ = phi_14059_;
                                                                                    let _e5127_ = phi_14060_;
                                                                                    phi_14065_ = select(_e5125_, 1000000000f, _e5127_);
                                                                                } else {
                                                                                    phi_14065_ = 0f;
                                                                                }
                                                                                let _e5130_ = phi_14065_;
                                                                                if _e4729_ {
                                                                                    let _e5131_ = (48u + _e197_);
                                                                                    if (_e5131_ < 256u) {
                                                                                    } else {
                                                                                        phi_36766_ = true;
                                                                                        break;
                                                                                    }
                                                                                    aux_scratch[_e5131_] = _e5130_;
                                                                                }
                                                                                workgroupBarrier();
                                                                                if _e4733_ {
                                                                                    let _e5134_ = (48u + _e197_);
                                                                                    let _e5135_ = (_e5134_ < 256u);
                                                                                    if _e5135_ {
                                                                                    } else {
                                                                                        phi_36766_ = true;
                                                                                        break;
                                                                                    }
                                                                                    let _e5137_ = aux_scratch[_e5134_];
                                                                                    if _e5135_ {
                                                                                    } else {
                                                                                        phi_36766_ = true;
                                                                                        break;
                                                                                    }
                                                                                    aux_scratch[_e5134_] = select(_e5137_, _e5130_, (_e5130_ > _e5137_));
                                                                                }
                                                                                workgroupBarrier();
                                                                                phi_14103_ = 64u;
                                                                                loop {
                                                                                    let _e5141_ = phi_14103_;
                                                                                    let _e5142_ = (_e5141_ > 0u);
                                                                                    if _e5142_ {
                                                                                        if (_e197_ < _e5141_) {
                                                                                            if _e4729_ {
                                                                                                let _e5144_ = (48u + _e197_);
                                                                                                let _e5145_ = (_e5144_ < 256u);
                                                                                                if _e5145_ {
                                                                                                } else {
                                                                                                    phi_36667_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                let _e5147_ = aux_scratch[_e5144_];
                                                                                                let _e5148_ = (_e5144_ + _e5141_);
                                                                                                if (_e5148_ < 256u) {
                                                                                                } else {
                                                                                                    phi_36667_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                let _e5151_ = aux_scratch[_e5148_];
                                                                                                if _e5145_ {
                                                                                                } else {
                                                                                                    phi_36667_ = true;
                                                                                                    break;
                                                                                                }
                                                                                                aux_scratch[_e5144_] = select(_e5147_, _e5151_, (_e5151_ > _e5147_));
                                                                                            }
                                                                                        }
                                                                                        workgroupBarrier();
                                                                                        phi_14104_ = (_e5141_ >> bitcast<u32>(1i));
                                                                                    } else {
                                                                                        phi_14104_ = u32();
                                                                                    }
                                                                                    let _e5157_ = phi_14104_;
                                                                                    phi_14103_ = _e5157_;
                                                                                    phi_36667_ = _e4834_;
                                                                                    if !(_e5142_) {
                                                                                        break;
                                                                                    }
                                                                                }
                                                                                let _e5160_ = phi_36667_;
                                                                                phi_36766_ = _e5160_;
                                                                                if _e5160_ {
                                                                                    break;
                                                                                }
                                                                                let _e5161_ = aux_scratch[48];
                                                                                workgroupBarrier();
                                                                                if (_e5161_ <= _e4762_) {
                                                                                    phi_14160_ = true;
                                                                                } else {
                                                                                    phi_14160_ = (_e4873_.member_1_ == 2i);
                                                                                }
                                                                                let _e5165_ = phi_14160_;
                                                                                if _e5165_ {
                                                                                    if _e4321_ {
                                                                                    } else {
                                                                                        phi_36766_ = true;
                                                                                        break;
                                                                                    }
                                                                                    scan_scratch[_e4320_] = _e4910_;
                                                                                    workgroupBarrier();
                                                                                    phi_14178_ = select(_e4844_, true, (_e5161_ <= 0.0000005f));
                                                                                } else {
                                                                                    phi_14178_ = _e4844_;
                                                                                }
                                                                                let _e5169_ = phi_14178_;
                                                                                phi_36768_ = _e5160_;
                                                                                phi_14180_ = select(_e4838_, _e4910_, _e5165_);
                                                                                phi_14181_ = (_e4840_ * 0.5f);
                                                                                phi_14182_ = select(_e4842_, true, _e5165_);
                                                                                phi_14183_ = _e5169_;
                                                                            }
                                                                            let _e5174_ = phi_36768_;
                                                                            let _e5176_ = phi_14180_;
                                                                            let _e5178_ = phi_14181_;
                                                                            let _e5180_ = phi_14182_;
                                                                            let _e5182_ = phi_14183_;
                                                                            phi_36767_ = _e5174_;
                                                                            phi_12678_ = _e5176_;
                                                                            phi_12680_ = _e5178_;
                                                                            phi_12682_ = _e5180_;
                                                                            phi_12684_ = _e5182_;
                                                                        } else {
                                                                            phi_36767_ = _e4834_;
                                                                            phi_12678_ = f32();
                                                                            phi_12680_ = f32();
                                                                            phi_12682_ = bool();
                                                                            phi_12684_ = bool();
                                                                        }
                                                                        let _e5184_ = phi_36767_;
                                                                        let _e5186_ = phi_12678_;
                                                                        let _e5188_ = phi_12680_;
                                                                        let _e5190_ = phi_12682_;
                                                                        let _e5192_ = phi_12684_;
                                                                        phi_36676_ = _e5184_;
                                                                        phi_12674_ = _e4871_;
                                                                        phi_12677_ = _e5186_;
                                                                        phi_12679_ = _e5188_;
                                                                        phi_12681_ = _e5190_;
                                                                        phi_12683_ = _e5192_;
                                                                        phi_36766_ = _e5184_;
                                                                        if !(_e4877_) {
                                                                            break;
                                                                        }
                                                                    }
                                                                    let _e5195_ = phi_36766_;
                                                                    phi_36772_ = _e5195_;
                                                                    if _e5195_ {
                                                                        break;
                                                                    }
                                                                    phi_36775_ = _e5195_;
                                                                    let _e11970_ = local_36_;
                                                                    phi_14184_ = _e11970_;
                                                                    let _e11973_ = local_37_;
                                                                    phi_14185_ = _e11973_;
                                                                }
                                                                let _e5197_ = phi_36775_;
                                                                let _e5199_ = phi_14184_;
                                                                let _e5201_ = phi_14185_;
                                                                phi_36774_ = _e5197_;
                                                                phi_14186_ = _e5199_;
                                                                phi_14187_ = _e5201_;
                                                            }
                                                            let _e5203_ = phi_36774_;
                                                            let _e5205_ = phi_14186_;
                                                            let _e5207_ = phi_14187_;
                                                            phi_36773_ = _e5203_;
                                                            phi_11177_ = _e5205_;
                                                            phi_11179_ = _e5207_;
                                                        } else {
                                                            phi_36773_ = _e4324_;
                                                            phi_11177_ = f32();
                                                            phi_11179_ = bool();
                                                        }
                                                        let _e5209_ = phi_36773_;
                                                        let _e5211_ = phi_11177_;
                                                        let _e5213_ = phi_11179_;
                                                        phi_36461_ = _e5209_;
                                                        phi_11173_ = _e4343_;
                                                        phi_11176_ = _e5211_;
                                                        phi_11178_ = _e5213_;
                                                        phi_36772_ = _e5209_;
                                                        if !(_e4348_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e5216_ = phi_36772_;
                                                    phi_36980_ = _e5216_;
                                                    if _e5216_ {
                                                        break;
                                                    }
                                                    if (_e197_ == 0u) {
                                                        phi_14201_ = _e4317_;
                                                    } else {
                                                        let _e5218_ = (_e4320_ - 1u);
                                                        if (_e5218_ < 1536u) {
                                                        } else {
                                                            phi_36980_ = true;
                                                            break;
                                                        }
                                                        let _e5221_ = scan_scratch[_e5218_];
                                                        phi_14201_ = _e5221_;
                                                    }
                                                    let _e5223_ = phi_14201_;
                                                    if _e2406_ {
                                                        let _e5227_ = (_e4285_ * _e4300_);
                                                        let _e5229_ = bitcast<u32>((_e4278_ * _e4311_));
                                                        let _e5230_ = (_e5229_ & 2139095040u);
                                                        if (_e5230_ == 2139095040u) {
                                                            phi_14229_ = true;
                                                        } else {
                                                            phi_14229_ = (_e5230_ == 0u);
                                                        }
                                                        let _e5234_ = phi_14229_;
                                                        let _e5239_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e5229_, 0u, _e5234_))) & 2147483647u));
                                                        if (_e5239_ != _e5239_) {
                                                            phi_27319_ = true;
                                                        } else {
                                                            phi_27319_ = (1000000000000000000f <= _e5239_);
                                                        }
                                                        let _e5243_ = phi_27319_;
                                                        let _e5246_ = (_e4258_ <= 0f);
                                                        let _e5247_ = select((_e4258_ * select(_e5239_, 1000000000000000000f, _e5243_)), 0f, _e5246_);
                                                        let _e5248_ = (1f + _e5247_);
                                                        let _e5251_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5248_)));
                                                        let _e5255_ = (_e5251_ + (_e5251_ * (1f - (_e5248_ * _e5251_))));
                                                        let _e5263_ = (1f - (_e4220_ * select((_e5255_ + (_e5255_ * (1f - (_e5248_ * _e5255_)))), 1f, (_e5247_ <= 0f))));
                                                        if (_e5263_ != _e5263_) {
                                                            phi_27379_ = true;
                                                        } else {
                                                            phi_27379_ = (0f >= _e5263_);
                                                        }
                                                        let _e5267_ = phi_27379_;
                                                        let _e5268_ = select(_e5263_, 0f, _e5267_);
                                                        if (_e5268_ != _e5268_) {
                                                            phi_27439_ = true;
                                                        } else {
                                                            phi_27439_ = (1f <= _e5268_);
                                                        }
                                                        let _e5272_ = phi_27439_;
                                                        let _e5273_ = select(_e5268_, 1f, _e5272_);
                                                        let _e5274_ = (_e4201_ * _e5223_);
                                                        let _e5275_ = (1f - _e4277_);
                                                        let _e5276_ = (_e4239_ * _e5275_);
                                                        let _e5278_ = (_e5227_ - (_e5276_ * _e5274_));
                                                        if (_e5278_ != _e5278_) {
                                                            phi_27499_ = true;
                                                        } else {
                                                            phi_27499_ = (-1f >= _e5278_);
                                                        }
                                                        let _e5282_ = phi_27499_;
                                                        let _e5283_ = select(_e5278_, -1f, _e5282_);
                                                        if (_e5283_ != _e5283_) {
                                                            phi_27559_ = true;
                                                        } else {
                                                            phi_27559_ = (1f <= _e5283_);
                                                        }
                                                        let _e5287_ = phi_27559_;
                                                        let _e5288_ = select(_e5283_, 1f, _e5287_);
                                                        let _e5297_ = select(0f, _e5278_, (bitcast<f32>((bitcast<u32>(_e5278_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                        let _e5298_ = (_e5297_ * 499.99997f);
                                                        if (_e5298_ != _e5298_) {
                                                            phi_27619_ = true;
                                                        } else {
                                                            phi_27619_ = (0f >= _e5298_);
                                                        }
                                                        let _e5302_ = phi_27619_;
                                                        let _e5303_ = select(_e5298_, 0f, _e5302_);
                                                        if (_e5303_ != _e5303_) {
                                                            phi_27679_ = true;
                                                        } else {
                                                            phi_27679_ = (1f <= _e5303_);
                                                        }
                                                        let _e5307_ = phi_27679_;
                                                        let _e5308_ = select(_e5303_, 1f, _e5307_);
                                                        let _e5314_ = (_e5297_ - 0.002f);
                                                        if (_e5314_ != _e5314_) {
                                                            phi_27739_ = true;
                                                        } else {
                                                            phi_27739_ = (0f >= _e5314_);
                                                        }
                                                        let _e5318_ = phi_27739_;
                                                        let _e5320_ = ((0.001f * (((_e5308_ * _e5308_) * _e5308_) * (2f - _e5308_))) + select(_e5314_, 0f, _e5318_));
                                                        let _e5324_ = (_e4277_ <= 0f);
                                                        let _e5331_ = bitcast<u32>((((1f - (_e4277_ * _e5273_)) * _e5274_) + (_e5273_ * select(((_e5275_ * _e5320_) + (_e4277_ * (_e5288_ * (1f - (0.33333334f * (_e5288_ * _e5288_)))))), _e5320_, _e5324_))));
                                                        let _e5332_ = (_e5331_ & 2139095040u);
                                                        if (_e5332_ == 2139095040u) {
                                                            phi_14678_ = true;
                                                        } else {
                                                            phi_14678_ = (_e5332_ == 0u);
                                                        }
                                                        let _e5336_ = phi_14678_;
                                                        let _e5339_ = bitcast<u32>(_e5227_);
                                                        let _e5340_ = (_e5339_ & 2139095040u);
                                                        if (_e5340_ == 2139095040u) {
                                                            phi_14726_ = true;
                                                        } else {
                                                            phi_14726_ = (_e5340_ == 0u);
                                                        }
                                                        let _e5344_ = phi_14726_;
                                                        let _e5347_ = bitcast<u32>(_e4311_);
                                                        let _e5348_ = (_e5347_ & 2139095040u);
                                                        if (_e5348_ == 2139095040u) {
                                                            phi_14742_ = true;
                                                        } else {
                                                            phi_14742_ = (_e5348_ == 0u);
                                                        }
                                                        let _e5352_ = phi_14742_;
                                                        let _e5355_ = bitcast<u32>(_e5223_);
                                                        let _e5356_ = (_e5355_ & 2139095040u);
                                                        if (_e5356_ == 2139095040u) {
                                                            phi_14758_ = true;
                                                        } else {
                                                            phi_14758_ = (_e5356_ == 0u);
                                                        }
                                                        let _e5360_ = phi_14758_;
                                                        let _e5364_ = bitcast<u32>((_e4278_ * bitcast<f32>(select(_e5347_, 0u, _e5352_))));
                                                        let _e5365_ = (_e5364_ & 2139095040u);
                                                        if (_e5365_ == 2139095040u) {
                                                            phi_14775_ = true;
                                                        } else {
                                                            phi_14775_ = (_e5365_ == 0u);
                                                        }
                                                        let _e5369_ = phi_14775_;
                                                        let _e5372_ = abs(bitcast<f32>(select(_e5364_, 0u, _e5369_)));
                                                        if (_e5372_ != _e5372_) {
                                                            phi_27799_ = true;
                                                        } else {
                                                            phi_27799_ = (1000000000000000000f <= _e5372_);
                                                        }
                                                        let _e5376_ = phi_27799_;
                                                        if _e5246_ {
                                                            phi_14788_ = 0f;
                                                        } else {
                                                            phi_14788_ = (_e4258_ * select(_e5372_, 1000000000000000000f, _e5376_));
                                                        }
                                                        let _e5380_ = phi_14788_;
                                                        if (_e5380_ <= 0f) {
                                                            phi_14805_ = 1f;
                                                        } else {
                                                            let _e5382_ = (1f + _e5380_);
                                                            let _e5385_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5382_)));
                                                            let _e5389_ = (_e5385_ + (_e5385_ * (1f - (_e5382_ * _e5385_))));
                                                            phi_14805_ = (_e5389_ + (_e5389_ * (1f - (_e5382_ * _e5389_))));
                                                        }
                                                        let _e5395_ = phi_14805_;
                                                        let _e5397_ = (1f - (_e4220_ * _e5395_));
                                                        let _e5398_ = bitcast<u32>(_e5397_);
                                                        let _e5399_ = (_e5398_ & 2139095040u);
                                                        let _e5400_ = (_e5399_ == 2139095040u);
                                                        if _e5400_ {
                                                            if ((_e5398_ & 2147483647u) > 2139095040u) {
                                                                phi_14852_ = f32();
                                                                phi_14853_ = true;
                                                            } else {
                                                                let _e5415_ = ((_e5398_ & 2147483648u) == 0u);
                                                                if _e5415_ {
                                                                    phi_14850_ = 1f;
                                                                } else {
                                                                    phi_14850_ = f32();
                                                                }
                                                                let _e5417_ = phi_14850_;
                                                                phi_14852_ = _e5417_;
                                                                phi_14853_ = select(true, false, _e5415_);
                                                            }
                                                            let _e5420_ = phi_14852_;
                                                            let _e5422_ = phi_14853_;
                                                            phi_14932_ = select(_e5420_, 0f, _e5422_);
                                                        } else {
                                                            if (_e5397_ < 0f) {
                                                                phi_14931_ = 0f;
                                                            } else {
                                                                if (_e5397_ > 1f) {
                                                                    phi_14930_ = 1f;
                                                                } else {
                                                                    if _e5400_ {
                                                                        phi_14924_ = true;
                                                                    } else {
                                                                        phi_14924_ = (_e5399_ == 0u);
                                                                    }
                                                                    let _e5405_ = phi_14924_;
                                                                    phi_14930_ = bitcast<f32>(select(_e5398_, 0u, _e5405_));
                                                                }
                                                                let _e5409_ = phi_14930_;
                                                                phi_14931_ = _e5409_;
                                                            }
                                                            let _e5411_ = phi_14931_;
                                                            phi_14932_ = _e5411_;
                                                        }
                                                        let _e5425_ = phi_14932_;
                                                        let _e5428_ = (bitcast<f32>(select(_e5339_, 0u, _e5344_)) - (_e5276_ * (_e4201_ * bitcast<f32>(select(_e5355_, 0u, _e5360_)))));
                                                        let _e5434_ = (select(0f, _e5428_, (bitcast<f32>((bitcast<u32>(_e5428_) & 2147483647u)) <= 340282350000000000000000000000000000000f)) * 499.99997f);
                                                        if (_e5434_ != _e5434_) {
                                                            phi_27814_ = true;
                                                        } else {
                                                            phi_27814_ = (0f >= _e5434_);
                                                        }
                                                        let _e5438_ = phi_27814_;
                                                        let _e5439_ = select(_e5434_, 0f, _e5438_);
                                                        if (_e5439_ != _e5439_) {
                                                            phi_27874_ = true;
                                                        } else {
                                                            phi_27874_ = (1f <= _e5439_);
                                                        }
                                                        let _e5443_ = phi_27874_;
                                                        let _e5444_ = select(_e5439_, 1f, _e5443_);
                                                        let _e5448_ = ((_e5444_ * _e5444_) * (3f - (2f * _e5444_)));
                                                        if (abs(_e5428_) < 1f) {
                                                            phi_15136_ = (1f - (_e5428_ * _e5428_));
                                                        } else {
                                                            phi_15136_ = 0f;
                                                        }
                                                        let _e5454_ = phi_15136_;
                                                        if _e5324_ {
                                                            phi_15144_ = _e5448_;
                                                        } else {
                                                            phi_15144_ = ((_e5275_ * _e5448_) + (_e4277_ * _e5454_));
                                                        }
                                                        let _e5459_ = phi_15144_;
                                                        let _e5466_ = (_e4201_ * ((1f - (_e4277_ * _e5425_)) - (((_e5425_ * _e4239_) * _e5275_) * _e5459_)));
                                                        let _e5467_ = bitcast<u32>(_e5466_);
                                                        let _e5468_ = (_e5467_ & 2139095040u);
                                                        let _e5469_ = (_e5468_ == 2139095040u);
                                                        if _e5469_ {
                                                            if ((_e5467_ & 2147483647u) > 2139095040u) {
                                                                phi_15201_ = f32();
                                                                phi_15202_ = true;
                                                            } else {
                                                                let _e5484_ = ((_e5467_ & 2147483648u) == 0u);
                                                                if _e5484_ {
                                                                    phi_15199_ = 0.999995f;
                                                                } else {
                                                                    phi_15199_ = f32();
                                                                }
                                                                let _e5486_ = phi_15199_;
                                                                phi_15201_ = _e5486_;
                                                                phi_15202_ = select(true, false, _e5484_);
                                                            }
                                                            let _e5489_ = phi_15201_;
                                                            let _e5491_ = phi_15202_;
                                                            if _e5491_ {
                                                                phi_15222_ = -0.999995f;
                                                            } else {
                                                                phi_15222_ = _e5489_;
                                                            }
                                                            let _e5493_ = phi_15222_;
                                                            phi_15281_ = _e5493_;
                                                        } else {
                                                            if (_e5466_ < -0.999995f) {
                                                                phi_15280_ = -0.999995f;
                                                            } else {
                                                                if (_e5466_ > 0.999995f) {
                                                                    phi_15279_ = 0.999995f;
                                                                } else {
                                                                    if _e5469_ {
                                                                        phi_15273_ = true;
                                                                    } else {
                                                                        phi_15273_ = (_e5468_ == 0u);
                                                                    }
                                                                    let _e5474_ = phi_15273_;
                                                                    phi_15279_ = bitcast<f32>(select(_e5467_, 0u, _e5474_));
                                                                }
                                                                let _e5478_ = phi_15279_;
                                                                phi_15280_ = _e5478_;
                                                            }
                                                            let _e5480_ = phi_15280_;
                                                            phi_15281_ = _e5480_;
                                                        }
                                                        let _e5495_ = phi_15281_;
                                                        phi_15284_ = type_29_(bitcast<f32>(select(_e5331_, 0u, _e5336_)), _e5495_);
                                                    } else {
                                                        let _e5225_ = local_29_;
                                                        phi_15284_ = type_29_(_e5225_, 1f);
                                                    }
                                                    let _e5498_ = phi_15284_;
                                                    if _e2406_ {
                                                        if ((bitcast<u32>(_e5498_.member) & 2139095040u) != 2139095040u) {
                                                            let _e5504_ = local_30_;
                                                            let _e5507_ = ((bitcast<u32>(_e5504_) & 2139095040u) != 2139095040u);
                                                            if _e5507_ {
                                                                let _e5508_ = abs(_e5498_.member);
                                                                let _e5510_ = local_31_;
                                                                let _e5511_ = abs(_e5510_);
                                                                let _e5513_ = select(_e5511_, _e5508_, (_e5508_ > _e5511_));
                                                                let _e5515_ = select(1f, _e5513_, (_e5513_ > 1f));
                                                                let _e5517_ = local_32_;
                                                                if ((bitcast<u32>(_e5515_) & 2139095040u) != 2139095040u) {
                                                                    let _e5523_ = (_e5515_ <= 0f);
                                                                    if _e5523_ {
                                                                        phi_15407_ = f32();
                                                                    } else {
                                                                        if (_e5515_ < 0.00001f) {
                                                                            let _e5555_ = (select(0.000000000000000000000001f, _e5515_, (_e5515_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                                                            if (_e5555_ != _e5555_) {
                                                                                phi_28054_ = true;
                                                                            } else {
                                                                                phi_28054_ = (0.000001f >= _e5555_);
                                                                            }
                                                                            let _e5559_ = phi_28054_;
                                                                            let _e5560_ = select(_e5555_, 0.000001f, _e5559_);
                                                                            let _e5563_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5560_)));
                                                                            let _e5567_ = (_e5563_ + (_e5563_ * (1f - (_e5560_ * _e5563_))));
                                                                            let _e5571_ = (_e5567_ + (_e5567_ * (1f - (_e5560_ * _e5567_))));
                                                                            let _e5577_ = bitcast<u32>(((_e5571_ + (_e5571_ * (1f - (_e5560_ * _e5571_)))) * 18446744000000000000f));
                                                                            let _e5578_ = (_e5577_ & 2139095040u);
                                                                            if (_e5578_ == 2139095040u) {
                                                                                phi_15365_ = true;
                                                                            } else {
                                                                                phi_15365_ = (_e5578_ == 0u);
                                                                            }
                                                                            let _e5582_ = phi_15365_;
                                                                            phi_15406_ = bitcast<f32>(select(_e5577_, 0u, _e5582_));
                                                                        } else {
                                                                            if (_e5515_ != _e5515_) {
                                                                                phi_27994_ = true;
                                                                            } else {
                                                                                phi_27994_ = (0.000001f >= _e5515_);
                                                                            }
                                                                            let _e5528_ = phi_27994_;
                                                                            let _e5529_ = select(_e5515_, 0.000001f, _e5528_);
                                                                            let _e5532_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5529_)));
                                                                            let _e5536_ = (_e5532_ + (_e5532_ * (1f - (_e5529_ * _e5532_))));
                                                                            let _e5540_ = (_e5536_ + (_e5536_ * (1f - (_e5529_ * _e5536_))));
                                                                            let _e5545_ = bitcast<u32>((_e5540_ + (_e5540_ * (1f - (_e5529_ * _e5540_)))));
                                                                            let _e5546_ = (_e5545_ & 2139095040u);
                                                                            if (_e5546_ == 2139095040u) {
                                                                                phi_15400_ = true;
                                                                            } else {
                                                                                phi_15400_ = (_e5546_ == 0u);
                                                                            }
                                                                            let _e5550_ = phi_15400_;
                                                                            phi_15406_ = bitcast<f32>(select(_e5545_, 0u, _e5550_));
                                                                        }
                                                                        let _e5586_ = phi_15406_;
                                                                        phi_15407_ = _e5586_;
                                                                    }
                                                                    let _e5588_ = phi_15407_;
                                                                    phi_15408_ = _e5588_;
                                                                    phi_15409_ = _e5523_;
                                                                } else {
                                                                    phi_15408_ = f32();
                                                                    phi_15409_ = true;
                                                                }
                                                                let _e5590_ = phi_15408_;
                                                                let _e5592_ = phi_15409_;
                                                                phi_15415_ = (abs((_e5498_.member - _e5517_)) * select(_e5590_, 0f, _e5592_));
                                                            } else {
                                                                phi_15415_ = f32();
                                                            }
                                                            let _e5596_ = phi_15415_;
                                                            phi_15417_ = _e5596_;
                                                            phi_15418_ = select(true, false, _e5507_);
                                                        } else {
                                                            phi_15417_ = f32();
                                                            phi_15418_ = true;
                                                        }
                                                        let _e5599_ = phi_15417_;
                                                        let _e5601_ = phi_15418_;
                                                        phi_15423_ = select(_e5599_, 1000000000f, _e5601_);
                                                    } else {
                                                        phi_15423_ = 0f;
                                                    }
                                                    let _e5604_ = phi_15423_;
                                                    let _e5605_ = (_e196_ < 128u);
                                                    if _e5605_ {
                                                        let _e5606_ = (48u + _e197_);
                                                        if (_e5606_ < 256u) {
                                                        } else {
                                                            phi_36980_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e5606_] = _e5604_;
                                                    }
                                                    workgroupBarrier();
                                                    if (_e196_ >= 128u) {
                                                        let _e5610_ = (48u + _e197_);
                                                        let _e5611_ = (_e5610_ < 256u);
                                                        if _e5611_ {
                                                        } else {
                                                            phi_36980_ = true;
                                                            break;
                                                        }
                                                        let _e5613_ = aux_scratch[_e5610_];
                                                        if _e5611_ {
                                                        } else {
                                                            phi_36980_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e5610_] = select(_e5613_, _e5604_, (_e5604_ > _e5613_));
                                                    }
                                                    workgroupBarrier();
                                                    phi_15461_ = 64u;
                                                    loop {
                                                        let _e5617_ = phi_15461_;
                                                        let _e5618_ = (_e5617_ > 0u);
                                                        if _e5618_ {
                                                            if (_e197_ < _e5617_) {
                                                                if _e5605_ {
                                                                    let _e5620_ = (48u + _e197_);
                                                                    let _e5621_ = (_e5620_ < 256u);
                                                                    if _e5621_ {
                                                                    } else {
                                                                        phi_36776_ = true;
                                                                        break;
                                                                    }
                                                                    let _e5623_ = aux_scratch[_e5620_];
                                                                    let _e5624_ = (_e5620_ + _e5617_);
                                                                    if (_e5624_ < 256u) {
                                                                    } else {
                                                                        phi_36776_ = true;
                                                                        break;
                                                                    }
                                                                    let _e5627_ = aux_scratch[_e5624_];
                                                                    if _e5621_ {
                                                                    } else {
                                                                        phi_36776_ = true;
                                                                        break;
                                                                    }
                                                                    aux_scratch[_e5620_] = select(_e5623_, _e5627_, (_e5627_ > _e5623_));
                                                                }
                                                            }
                                                            workgroupBarrier();
                                                            phi_15462_ = (_e5617_ >> bitcast<u32>(1i));
                                                        } else {
                                                            phi_15462_ = u32();
                                                        }
                                                        let _e5633_ = phi_15462_;
                                                        phi_15461_ = _e5633_;
                                                        phi_36776_ = _e5216_;
                                                        if !(_e5618_) {
                                                            break;
                                                        }
                                                    }
                                                    let _e5636_ = phi_36776_;
                                                    phi_36980_ = _e5636_;
                                                    if _e5636_ {
                                                        break;
                                                    }
                                                    let _e5638_ = aux_scratch[48];
                                                    workgroupBarrier();
                                                    if (_e5638_ > 0.0000005f) {
                                                        if (_e196_ < 2u) {
                                                            let _e5693_ = (_e196_ == 1u);
                                                            let _e5694_ = select(0u, 2u, _e5693_);
                                                            if _e5693_ {
                                                                phi_15541_ = (20u + _e4135_.member_1_);
                                                            } else {
                                                                phi_15541_ = (16u + _e4135_.member_1_);
                                                            }
                                                            let _e5699_ = phi_15541_;
                                                            let _e5700_ = (_e5699_ < 256u);
                                                            if _e5700_ {
                                                            } else {
                                                                phi_36980_ = true;
                                                                break;
                                                            }
                                                            let _e5702_ = aux_scratch[_e5699_];
                                                            phi_15551_ = type_15_(0u, _e2404_);
                                                            phi_15554_ = _e5702_;
                                                            loop {
                                                                let _e5705_ = phi_15551_;
                                                                let _e5707_ = phi_15554_;
                                                                local_33_ = _e5707_;
                                                                if (_e5705_.member < _e5705_.member_1_) {
                                                                    phi_15552_ = type_15_((_e5705_.member + 1u), _e5705_.member_1_);
                                                                    phi_15577_ = type_15_(1u, _e5705_.member);
                                                                } else {
                                                                    phi_15552_ = _e5705_;
                                                                    phi_15577_ = type_15_(0u, u32());
                                                                }
                                                                let _e5720_ = phi_15552_;
                                                                let _e5722_ = phi_15577_;
                                                                let _e5726_ = (bitcast<i32>(_e5722_.member) != 0i);
                                                                if _e5726_ {
                                                                    let _e5728_ = (_e4135_.member_1_ * 4u);
                                                                    let _e5731_ = ((_e5722_.member_1_ & 127u) & 127u);
                                                                    let _e5732_ = (((_e5728_ + _e5694_) * 128u) + _e5731_);
                                                                    if (_e5732_ < 2048u) {
                                                                    } else {
                                                                        phi_36929_ = true;
                                                                        break;
                                                                    }
                                                                    let _e5735_ = lane_buf[_e5732_];
                                                                    let _e5738_ = (((_e5728_ + (_e5694_ + 1u)) * 128u) + _e5731_);
                                                                    let _e5739_ = (_e5738_ < 2048u);
                                                                    if _e5739_ {
                                                                    } else {
                                                                        phi_36929_ = true;
                                                                        break;
                                                                    }
                                                                    let _e5741_ = lane_buf[_e5738_];
                                                                    let _e5744_ = bitcast<u32>((_e4278_ * _e5741_));
                                                                    let _e5745_ = (_e5744_ & 2139095040u);
                                                                    if (_e5745_ == 2139095040u) {
                                                                        phi_15622_ = true;
                                                                    } else {
                                                                        phi_15622_ = (_e5745_ == 0u);
                                                                    }
                                                                    let _e5749_ = phi_15622_;
                                                                    let _e5754_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e5744_, 0u, _e5749_))) & 2147483647u));
                                                                    if (_e5754_ != _e5754_) {
                                                                        phi_28114_ = true;
                                                                    } else {
                                                                        phi_28114_ = (1000000000000000000f <= _e5754_);
                                                                    }
                                                                    let _e5758_ = phi_28114_;
                                                                    let _e5762_ = select((_e4258_ * select(_e5754_, 1000000000000000000f, _e5758_)), 0f, (_e4258_ <= 0f));
                                                                    let _e5763_ = (1f + _e5762_);
                                                                    let _e5766_ = bitcast<f32>((2129859011u - bitcast<u32>(_e5763_)));
                                                                    let _e5770_ = (_e5766_ + (_e5766_ * (1f - (_e5763_ * _e5766_))));
                                                                    let _e5778_ = (1f - (_e4220_ * select((_e5770_ + (_e5770_ * (1f - (_e5763_ * _e5770_)))), 1f, (_e5762_ <= 0f))));
                                                                    if (_e5778_ != _e5778_) {
                                                                        phi_28174_ = true;
                                                                    } else {
                                                                        phi_28174_ = (0f >= _e5778_);
                                                                    }
                                                                    let _e5782_ = phi_28174_;
                                                                    let _e5783_ = select(_e5778_, 0f, _e5782_);
                                                                    if (_e5783_ != _e5783_) {
                                                                        phi_28234_ = true;
                                                                    } else {
                                                                        phi_28234_ = (1f <= _e5783_);
                                                                    }
                                                                    let _e5787_ = phi_28234_;
                                                                    let _e5788_ = select(_e5783_, 1f, _e5787_);
                                                                    let _e5789_ = (_e4201_ * _e5707_);
                                                                    let _e5790_ = (1f - _e4277_);
                                                                    let _e5793_ = ((_e4285_ * _e5735_) - ((_e4239_ * _e5790_) * _e5789_));
                                                                    if (_e5793_ != _e5793_) {
                                                                        phi_28294_ = true;
                                                                    } else {
                                                                        phi_28294_ = (-1f >= _e5793_);
                                                                    }
                                                                    let _e5797_ = phi_28294_;
                                                                    let _e5798_ = select(_e5793_, -1f, _e5797_);
                                                                    if (_e5798_ != _e5798_) {
                                                                        phi_28354_ = true;
                                                                    } else {
                                                                        phi_28354_ = (1f <= _e5798_);
                                                                    }
                                                                    let _e5802_ = phi_28354_;
                                                                    let _e5803_ = select(_e5798_, 1f, _e5802_);
                                                                    let _e5812_ = select(0f, _e5793_, (bitcast<f32>((bitcast<u32>(_e5793_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                                    let _e5813_ = (_e5812_ * 499.99997f);
                                                                    if (_e5813_ != _e5813_) {
                                                                        phi_28414_ = true;
                                                                    } else {
                                                                        phi_28414_ = (0f >= _e5813_);
                                                                    }
                                                                    let _e5817_ = phi_28414_;
                                                                    let _e5818_ = select(_e5813_, 0f, _e5817_);
                                                                    if (_e5818_ != _e5818_) {
                                                                        phi_28474_ = true;
                                                                    } else {
                                                                        phi_28474_ = (1f <= _e5818_);
                                                                    }
                                                                    let _e5822_ = phi_28474_;
                                                                    let _e5823_ = select(_e5818_, 1f, _e5822_);
                                                                    let _e5829_ = (_e5812_ - 0.002f);
                                                                    if (_e5829_ != _e5829_) {
                                                                        phi_28534_ = true;
                                                                    } else {
                                                                        phi_28534_ = (0f >= _e5829_);
                                                                    }
                                                                    let _e5833_ = phi_28534_;
                                                                    let _e5835_ = ((0.001f * (((_e5823_ * _e5823_) * _e5823_) * (2f - _e5823_))) + select(_e5829_, 0f, _e5833_));
                                                                    let _e5846_ = bitcast<u32>((((1f - (_e4277_ * _e5788_)) * _e5789_) + (_e5788_ * select(((_e5790_ * _e5835_) + (_e4277_ * (_e5803_ * (1f - (0.33333334f * (_e5803_ * _e5803_)))))), _e5835_, (_e4277_ <= 0f)))));
                                                                    let _e5847_ = (_e5846_ & 2139095040u);
                                                                    if (_e5847_ == 2139095040u) {
                                                                        phi_16071_ = true;
                                                                    } else {
                                                                        phi_16071_ = (_e5847_ == 0u);
                                                                    }
                                                                    let _e5851_ = phi_16071_;
                                                                    let _e5853_ = bitcast<f32>(select(_e5846_, 0u, _e5851_));
                                                                    if _e5739_ {
                                                                    } else {
                                                                        phi_36929_ = true;
                                                                        break;
                                                                    }
                                                                    lane_buf[_e5738_] = _e5853_;
                                                                    phi_15555_ = _e5853_;
                                                                } else {
                                                                    phi_15555_ = f32();
                                                                }
                                                                let _e5855_ = phi_15555_;
                                                                phi_15551_ = _e5720_;
                                                                phi_15554_ = _e5855_;
                                                                phi_36929_ = _e5636_;
                                                                if !(_e5726_) {
                                                                    break;
                                                                }
                                                            }
                                                            let _e5858_ = phi_36929_;
                                                            phi_36980_ = _e5858_;
                                                            if _e5858_ {
                                                                break;
                                                            }
                                                            let _e5860_ = local_33_;
                                                            let _e5861_ = bitcast<u32>(_e5860_);
                                                            let _e5862_ = (_e5861_ & 2139095040u);
                                                            if (_e5862_ == 2139095040u) {
                                                                phi_16129_ = true;
                                                            } else {
                                                                phi_16129_ = (_e5862_ == 0u);
                                                            }
                                                            let _e5866_ = phi_16129_;
                                                            if _e5700_ {
                                                            } else {
                                                                phi_36980_ = true;
                                                                break;
                                                            }
                                                            aux_scratch[_e5699_] = bitcast<f32>(select(_e5861_, 0u, _e5866_));
                                                            phi_36992_ = _e5858_;
                                                        } else {
                                                            phi_36992_ = _e5636_;
                                                        }
                                                        let _e5870_ = phi_36992_;
                                                        workgroupBarrier();
                                                        phi_36983_ = _e5870_;
                                                    } else {
                                                        workgroupBarrier();
                                                        if _e5605_ {
                                                            if _e2406_ {
                                                                let _e5640_ = (512u + _e197_);
                                                                if (_e5640_ < 1536u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                let _e5643_ = scan_scratch[_e5640_];
                                                                let _e5644_ = (_e4135_.member_1_ * 4u);
                                                                let _e5647_ = (_e197_ & 127u);
                                                                let _e5648_ = (((_e5644_ + 1u) * 128u) + _e5647_);
                                                                if (_e5648_ < 2048u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e5648_] = _e5643_;
                                                                let _e5651_ = (640u + _e197_);
                                                                if (_e5651_ < 1536u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                let _e5654_ = scan_scratch[_e5651_];
                                                                let _e5657_ = (((_e5644_ + 3u) * 128u) + _e5647_);
                                                                if (_e5657_ < 2048u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                lane_buf[_e5657_] = _e5654_;
                                                            }
                                                        }
                                                        if _e1827_ {
                                                            if _e2405_ {
                                                                let _e5661_ = ((_e2404_ - 1u) & 127u);
                                                                let _e5662_ = (512u + _e5661_);
                                                                if (_e5662_ < 1536u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                let _e5665_ = scan_scratch[_e5662_];
                                                                let _e5666_ = bitcast<u32>(_e5665_);
                                                                let _e5667_ = (_e5666_ & 2139095040u);
                                                                if (_e5667_ == 2139095040u) {
                                                                    phi_16214_ = true;
                                                                } else {
                                                                    phi_16214_ = (_e5667_ == 0u);
                                                                }
                                                                let _e5671_ = phi_16214_;
                                                                let _e5674_ = (16u + _e4135_.member_1_);
                                                                if (_e5674_ < 256u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e5674_] = bitcast<f32>(select(_e5666_, 0u, _e5671_));
                                                                let _e5677_ = (640u + _e5661_);
                                                                if (_e5677_ < 1536u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                let _e5680_ = scan_scratch[_e5677_];
                                                                let _e5681_ = bitcast<u32>(_e5680_);
                                                                let _e5682_ = (_e5681_ & 2139095040u);
                                                                if (_e5682_ == 2139095040u) {
                                                                    phi_16245_ = true;
                                                                } else {
                                                                    phi_16245_ = (_e5682_ == 0u);
                                                                }
                                                                let _e5686_ = phi_16245_;
                                                                let _e5689_ = (20u + _e4135_.member_1_);
                                                                if (_e5689_ < 256u) {
                                                                } else {
                                                                    phi_36980_ = true;
                                                                    break;
                                                                }
                                                                aux_scratch[_e5689_] = bitcast<f32>(select(_e5681_, 0u, _e5686_));
                                                            }
                                                        }
                                                        workgroupBarrier();
                                                        phi_36983_ = _e5636_;
                                                    }
                                                    let _e5872_ = phi_36983_;
                                                    phi_36982_ = _e5872_;
                                                }
                                                let _e5874_ = phi_36982_;
                                                phi_36981_ = _e5874_;
                                            } else {
                                                phi_36981_ = _e4118_;
                                            }
                                            let _e5876_ = phi_36981_;
                                            phi_36489_ = _e5876_;
                                            phi_10028_ = _e4133_;
                                            phi_36980_ = _e5876_;
                                            if !(_e4139_) {
                                                break;
                                            }
                                        }
                                        let _e5879_ = phi_36980_;
                                        phi_38285_ = _e5879_;
                                        if _e5879_ {
                                            break;
                                        }
                                        phi_36999_ = _e5879_;
                                    } else {
                                        if (_e196_ < 8u) {
                                            let _e3785_ = (_e196_ >> bitcast<u32>(1i));
                                            let _e3787_ = ((_e196_ & 1u) != 0u);
                                            let _e3788_ = (_e3785_ & 3u);
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16281_ = _e437_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16281_ = _e437_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16281_ = _e437_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16281_ = _e437_.w;
                                                    break;
                                                }
                                            }
                                            let _e3795_ = phi_16281_;
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16292_ = _e395_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16292_ = _e395_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16292_ = _e395_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16292_ = _e395_.w;
                                                    break;
                                                }
                                            }
                                            let _e3802_ = phi_16292_;
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16303_ = _e402_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16303_ = _e402_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16303_ = _e402_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16303_ = _e402_.w;
                                                    break;
                                                }
                                            }
                                            let _e3809_ = phi_16303_;
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16314_ = _e409_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16314_ = _e409_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16314_ = _e409_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16314_ = _e409_.w;
                                                    break;
                                                }
                                            }
                                            let _e3816_ = phi_16314_;
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16325_ = _e416_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16325_ = _e416_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16325_ = _e416_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16325_ = _e416_.w;
                                                    break;
                                                }
                                            }
                                            let _e3823_ = phi_16325_;
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16336_ = _e423_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16336_ = _e423_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16336_ = _e423_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16336_ = _e423_.w;
                                                    break;
                                                }
                                            }
                                            let _e3830_ = phi_16336_;
                                            let _e3831_ = bitcast<u32>(_e3802_);
                                            let _e3832_ = (_e3831_ & 2139095040u);
                                            if (_e3832_ != 2139095040u) {
                                                if (_e3802_ <= 0f) {
                                                    phi_16520_ = 0f;
                                                } else {
                                                    if (_e3802_ >= 0.999995f) {
                                                        phi_16519_ = 0.999995f;
                                                    } else {
                                                        if (_e3832_ == 2139095040u) {
                                                            phi_16513_ = true;
                                                        } else {
                                                            phi_16513_ = (_e3832_ == 0u);
                                                        }
                                                        let _e3841_ = phi_16513_;
                                                        phi_16519_ = bitcast<f32>(select(_e3831_, 0u, _e3841_));
                                                    }
                                                    let _e3845_ = phi_16519_;
                                                    phi_16520_ = _e3845_;
                                                }
                                                let _e3847_ = phi_16520_;
                                                phi_16527_ = _e3847_;
                                            } else {
                                                phi_16527_ = select(0f, 0.999995f, (_e3831_ == 2139095040u));
                                            }
                                            let _e3849_ = phi_16527_;
                                            let _e3850_ = bitcast<u32>(_e3809_);
                                            let _e3851_ = (_e3850_ & 2139095040u);
                                            if (_e3851_ != 2139095040u) {
                                                if (_e3809_ <= 0f) {
                                                    phi_16559_ = 0f;
                                                } else {
                                                    if (_e3809_ >= 1f) {
                                                        phi_16558_ = 1f;
                                                    } else {
                                                        if (_e3851_ == 2139095040u) {
                                                            phi_16552_ = true;
                                                        } else {
                                                            phi_16552_ = (_e3851_ == 0u);
                                                        }
                                                        let _e3860_ = phi_16552_;
                                                        phi_16558_ = bitcast<f32>(select(_e3850_, 0u, _e3860_));
                                                    }
                                                    let _e3864_ = phi_16558_;
                                                    phi_16559_ = _e3864_;
                                                }
                                                let _e3866_ = phi_16559_;
                                                phi_16566_ = _e3866_;
                                            } else {
                                                phi_16566_ = select(0f, 1f, (_e3850_ == 2139095040u));
                                            }
                                            let _e3868_ = phi_16566_;
                                            let _e3869_ = bitcast<u32>(_e3816_);
                                            let _e3870_ = (_e3869_ & 2139095040u);
                                            if (_e3870_ != 2139095040u) {
                                                if (_e3816_ <= 0f) {
                                                    phi_16598_ = 0f;
                                                } else {
                                                    if (_e3816_ >= 1f) {
                                                        phi_16597_ = 1f;
                                                    } else {
                                                        if (_e3870_ == 2139095040u) {
                                                            phi_16591_ = true;
                                                        } else {
                                                            phi_16591_ = (_e3870_ == 0u);
                                                        }
                                                        let _e3879_ = phi_16591_;
                                                        phi_16597_ = bitcast<f32>(select(_e3869_, 0u, _e3879_));
                                                    }
                                                    let _e3883_ = phi_16597_;
                                                    phi_16598_ = _e3883_;
                                                }
                                                let _e3885_ = phi_16598_;
                                                phi_16605_ = _e3885_;
                                            } else {
                                                phi_16605_ = select(0f, 1f, (_e3869_ == 2139095040u));
                                            }
                                            let _e3887_ = phi_16605_;
                                            let _e3888_ = bitcast<u32>(_e3823_);
                                            let _e3889_ = (_e3888_ & 2139095040u);
                                            if (_e3889_ != 2139095040u) {
                                                if (_e3823_ <= 0f) {
                                                    phi_16637_ = 0f;
                                                } else {
                                                    if (_e3823_ >= 64f) {
                                                        phi_16636_ = 64f;
                                                    } else {
                                                        if (_e3889_ == 2139095040u) {
                                                            phi_16630_ = true;
                                                        } else {
                                                            phi_16630_ = (_e3889_ == 0u);
                                                        }
                                                        let _e3898_ = phi_16630_;
                                                        phi_16636_ = bitcast<f32>(select(_e3888_, 0u, _e3898_));
                                                    }
                                                    let _e3902_ = phi_16636_;
                                                    phi_16637_ = _e3902_;
                                                }
                                                let _e3904_ = phi_16637_;
                                                phi_16644_ = _e3904_;
                                            } else {
                                                phi_16644_ = select(0f, 64f, (_e3888_ == 2139095040u));
                                            }
                                            let _e3906_ = phi_16644_;
                                            let _e3907_ = bitcast<u32>(_e3830_);
                                            let _e3908_ = (_e3907_ & 2139095040u);
                                            if (_e3908_ != 2139095040u) {
                                                if (_e3830_ <= 0f) {
                                                    phi_16676_ = 0f;
                                                } else {
                                                    if (_e3830_ >= 1f) {
                                                        phi_16675_ = 1f;
                                                    } else {
                                                        if (_e3908_ == 2139095040u) {
                                                            phi_16669_ = true;
                                                        } else {
                                                            phi_16669_ = (_e3908_ == 0u);
                                                        }
                                                        let _e3917_ = phi_16669_;
                                                        phi_16675_ = bitcast<f32>(select(_e3907_, 0u, _e3917_));
                                                    }
                                                    let _e3921_ = phi_16675_;
                                                    phi_16676_ = _e3921_;
                                                }
                                                let _e3923_ = phi_16676_;
                                                phi_16683_ = _e3923_;
                                            } else {
                                                phi_16683_ = select(0f, 1f, (_e3907_ == 2139095040u));
                                            }
                                            let _e3925_ = phi_16683_;
                                            switch bitcast<i32>(_e3788_) {
                                                case 0: {
                                                    phi_16800_ = _e451_.x;
                                                    break;
                                                }
                                                case 1: {
                                                    phi_16800_ = _e451_.y;
                                                    break;
                                                }
                                                case 2: {
                                                    phi_16800_ = _e451_.z;
                                                    break;
                                                }
                                                default: {
                                                    phi_16800_ = _e451_.w;
                                                    break;
                                                }
                                            }
                                            let _e3933_ = phi_16800_;
                                            if (_e3887_ == 0f) {
                                                phi_17300_ = true;
                                            } else {
                                                phi_17300_ = (_e3925_ >= 1f);
                                            }
                                            let _e3937_ = phi_17300_;
                                            if _e3937_ {
                                                phi_37001_ = _e3782_;
                                            } else {
                                                let _e3938_ = select(0u, 2u, _e3787_);
                                                if _e3787_ {
                                                    phi_17326_ = (20u + _e3785_);
                                                } else {
                                                    phi_17326_ = (16u + _e3785_);
                                                }
                                                let _e3943_ = phi_17326_;
                                                let _e3944_ = (_e3943_ < 256u);
                                                if _e3944_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                let _e3946_ = aux_scratch[_e3943_];
                                                phi_17336_ = type_15_(0u, _e2404_);
                                                phi_17339_ = _e3946_;
                                                loop {
                                                    let _e3949_ = phi_17336_;
                                                    let _e3951_ = phi_17339_;
                                                    local_26_ = _e3951_;
                                                    if (_e3949_.member < _e3949_.member_1_) {
                                                        phi_17337_ = type_15_((_e3949_.member + 1u), _e3949_.member_1_);
                                                        phi_17362_ = type_15_(1u, _e3949_.member);
                                                    } else {
                                                        phi_17337_ = _e3949_;
                                                        phi_17362_ = type_15_(0u, u32());
                                                    }
                                                    let _e3964_ = phi_17337_;
                                                    let _e3966_ = phi_17362_;
                                                    let _e3970_ = (bitcast<i32>(_e3966_.member) != 0i);
                                                    if _e3970_ {
                                                        let _e3972_ = (_e3785_ * 4u);
                                                        let _e3975_ = ((_e3966_.member_1_ & 127u) & 127u);
                                                        let _e3976_ = (((_e3972_ + _e3938_) * 128u) + _e3975_);
                                                        if (_e3976_ < 2048u) {
                                                        } else {
                                                            phi_36356_ = true;
                                                            break;
                                                        }
                                                        let _e3979_ = lane_buf[_e3976_];
                                                        let _e3982_ = (((_e3972_ + (_e3938_ + 1u)) * 128u) + _e3975_);
                                                        let _e3983_ = (_e3982_ < 2048u);
                                                        if _e3983_ {
                                                        } else {
                                                            phi_36356_ = true;
                                                            break;
                                                        }
                                                        let _e3985_ = lane_buf[_e3982_];
                                                        let _e3988_ = bitcast<u32>(((_e3795_ + _e3795_) * _e3985_));
                                                        let _e3989_ = (_e3988_ & 2139095040u);
                                                        if (_e3989_ == 2139095040u) {
                                                            phi_17407_ = true;
                                                        } else {
                                                            phi_17407_ = (_e3989_ == 0u);
                                                        }
                                                        let _e3993_ = phi_17407_;
                                                        let _e3998_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e3988_, 0u, _e3993_))) & 2147483647u));
                                                        if (_e3998_ != _e3998_) {
                                                            phi_25249_ = true;
                                                        } else {
                                                            phi_25249_ = (1000000000000000000f <= _e3998_);
                                                        }
                                                        let _e4002_ = phi_25249_;
                                                        let _e4006_ = select((_e3906_ * select(_e3998_, 1000000000000000000f, _e4002_)), 0f, (_e3906_ <= 0f));
                                                        let _e4007_ = (1f + _e4006_);
                                                        let _e4010_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4007_)));
                                                        let _e4014_ = (_e4010_ + (_e4010_ * (1f - (_e4007_ * _e4010_))));
                                                        let _e4022_ = (1f - (_e3868_ * select((_e4014_ + (_e4014_ * (1f - (_e4007_ * _e4014_)))), 1f, (_e4006_ <= 0f))));
                                                        if (_e4022_ != _e4022_) {
                                                            phi_25309_ = true;
                                                        } else {
                                                            phi_25309_ = (0f >= _e4022_);
                                                        }
                                                        let _e4026_ = phi_25309_;
                                                        let _e4027_ = select(_e4022_, 0f, _e4026_);
                                                        if (_e4027_ != _e4027_) {
                                                            phi_25369_ = true;
                                                        } else {
                                                            phi_25369_ = (1f <= _e4027_);
                                                        }
                                                        let _e4031_ = phi_25369_;
                                                        let _e4032_ = select(_e4027_, 1f, _e4031_);
                                                        let _e4033_ = (_e3849_ * _e3951_);
                                                        let _e4034_ = (1f - _e3925_);
                                                        let _e4037_ = ((_e3933_ * _e3979_) - ((_e3887_ * _e4034_) * _e4033_));
                                                        if (_e4037_ != _e4037_) {
                                                            phi_25429_ = true;
                                                        } else {
                                                            phi_25429_ = (-1f >= _e4037_);
                                                        }
                                                        let _e4041_ = phi_25429_;
                                                        let _e4042_ = select(_e4037_, -1f, _e4041_);
                                                        if (_e4042_ != _e4042_) {
                                                            phi_25489_ = true;
                                                        } else {
                                                            phi_25489_ = (1f <= _e4042_);
                                                        }
                                                        let _e4046_ = phi_25489_;
                                                        let _e4047_ = select(_e4042_, 1f, _e4046_);
                                                        let _e4056_ = select(0f, _e4037_, (bitcast<f32>((bitcast<u32>(_e4037_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                        let _e4057_ = (_e4056_ * 499.99997f);
                                                        if (_e4057_ != _e4057_) {
                                                            phi_25549_ = true;
                                                        } else {
                                                            phi_25549_ = (0f >= _e4057_);
                                                        }
                                                        let _e4061_ = phi_25549_;
                                                        let _e4062_ = select(_e4057_, 0f, _e4061_);
                                                        if (_e4062_ != _e4062_) {
                                                            phi_25609_ = true;
                                                        } else {
                                                            phi_25609_ = (1f <= _e4062_);
                                                        }
                                                        let _e4066_ = phi_25609_;
                                                        let _e4067_ = select(_e4062_, 1f, _e4066_);
                                                        let _e4073_ = (_e4056_ - 0.002f);
                                                        if (_e4073_ != _e4073_) {
                                                            phi_25669_ = true;
                                                        } else {
                                                            phi_25669_ = (0f >= _e4073_);
                                                        }
                                                        let _e4077_ = phi_25669_;
                                                        let _e4079_ = ((0.001f * (((_e4067_ * _e4067_) * _e4067_) * (2f - _e4067_))) + select(_e4073_, 0f, _e4077_));
                                                        let _e4090_ = bitcast<u32>((((1f - (_e3925_ * _e4032_)) * _e4033_) + (_e4032_ * select(((_e4034_ * _e4079_) + (_e3925_ * (_e4047_ * (1f - (0.33333334f * (_e4047_ * _e4047_)))))), _e4079_, (_e3925_ <= 0f)))));
                                                        let _e4091_ = (_e4090_ & 2139095040u);
                                                        if (_e4091_ == 2139095040u) {
                                                            phi_17856_ = true;
                                                        } else {
                                                            phi_17856_ = (_e4091_ == 0u);
                                                        }
                                                        let _e4095_ = phi_17856_;
                                                        let _e4097_ = bitcast<f32>(select(_e4090_, 0u, _e4095_));
                                                        if _e3983_ {
                                                        } else {
                                                            phi_36356_ = true;
                                                            break;
                                                        }
                                                        lane_buf[_e3982_] = _e4097_;
                                                        phi_17340_ = _e4097_;
                                                    } else {
                                                        phi_17340_ = f32();
                                                    }
                                                    let _e4099_ = phi_17340_;
                                                    phi_17336_ = _e3964_;
                                                    phi_17339_ = _e4099_;
                                                    phi_36356_ = _e3782_;
                                                    if !(_e3970_) {
                                                        break;
                                                    }
                                                }
                                                let _e4102_ = phi_36356_;
                                                phi_38285_ = _e4102_;
                                                if _e4102_ {
                                                    break;
                                                }
                                                let _e4104_ = local_26_;
                                                let _e4105_ = bitcast<u32>(_e4104_);
                                                let _e4106_ = (_e4105_ & 2139095040u);
                                                if (_e4106_ == 2139095040u) {
                                                    phi_17914_ = true;
                                                } else {
                                                    phi_17914_ = (_e4106_ == 0u);
                                                }
                                                let _e4110_ = phi_17914_;
                                                if _e3944_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                aux_scratch[_e3943_] = bitcast<f32>(select(_e4105_, 0u, _e4110_));
                                                phi_37001_ = _e4102_;
                                            }
                                            let _e4114_ = phi_37001_;
                                            phi_37000_ = _e4114_;
                                        } else {
                                            phi_37000_ = _e3782_;
                                        }
                                        let _e4116_ = phi_37000_;
                                        workgroupBarrier();
                                        phi_36999_ = _e4116_;
                                    }
                                    let _e5881_ = phi_36999_;
                                    let _e5882_ = (_e196_ < 128u);
                                    if _e5882_ {
                                        if _e2406_ {
                                            local_13_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_14_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_15_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_16_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            phi_17951_ = type_15_(0u, 4u);
                                            loop {
                                                let _e5884_ = phi_17951_;
                                                if (_e5884_.member < _e5884_.member_1_) {
                                                    phi_17952_ = type_15_((_e5884_.member + 1u), _e5884_.member_1_);
                                                    phi_17975_ = type_15_(1u, _e5884_.member);
                                                } else {
                                                    phi_17952_ = _e5884_;
                                                    phi_17975_ = type_15_(0u, u32());
                                                }
                                                let _e5897_ = phi_17952_;
                                                let _e5899_ = phi_17975_;
                                                let _e5903_ = (bitcast<i32>(_e5899_.member) != 0i);
                                                if _e5903_ {
                                                    let _e5904_ = (_e5899_.member_1_ * 4u);
                                                    let _e5906_ = (_e197_ & 127u);
                                                    let _e5907_ = ((_e5899_.member_1_ * 512u) + _e5906_);
                                                    if (_e5907_ < 2048u) {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    let _e5910_ = lane_buf[_e5907_];
                                                    let _e5911_ = (_e5899_.member_1_ < 4u);
                                                    if _e5911_ {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    local_13_[_e5899_.member_1_] = _e5910_;
                                                    let _e5915_ = (((_e5904_ + 1u) * 128u) + _e5906_);
                                                    if (_e5915_ < 2048u) {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    let _e5918_ = lane_buf[_e5915_];
                                                    if _e5911_ {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    local_15_[_e5899_.member_1_] = _e5918_;
                                                    let _e5922_ = (((_e5904_ + 2u) * 128u) + _e5906_);
                                                    if (_e5922_ < 2048u) {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    let _e5925_ = lane_buf[_e5922_];
                                                    if _e5911_ {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    local_14_[_e5899_.member_1_] = _e5925_;
                                                    let _e5929_ = (((_e5904_ + 3u) * 128u) + _e5906_);
                                                    if (_e5929_ < 2048u) {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    let _e5932_ = lane_buf[_e5929_];
                                                    if _e5911_ {
                                                    } else {
                                                        phi_36995_ = true;
                                                        break;
                                                    }
                                                    local_16_[_e5899_.member_1_] = _e5932_;
                                                }
                                                phi_17951_ = _e5897_;
                                                phi_36995_ = _e5881_;
                                                if !(_e5903_) {
                                                    break;
                                                }
                                            }
                                            let _e5936_ = phi_36995_;
                                            phi_38285_ = _e5936_;
                                            if _e5936_ {
                                                break;
                                            }
                                            let _e5937_ = local_13_;
                                            let _e5944_ = local_14_;
                                            let _e5951_ = local_15_;
                                            let _e5956_ = local_16_;
                                            let _e5965_ = (0.5f * (_e5951_[0] + _e5956_[0]));
                                            let _e5966_ = (0.5f * (_e5951_[1] + _e5956_[1]));
                                            let _e5967_ = (0.5f * (_e5951_[2] + _e5956_[2]));
                                            let _e5968_ = (0.5f * (_e5951_[3] + _e5956_[3]));
                                            let _e5974_ = (_e5951_[0] + (_e548_ * (_e5965_ - _e5951_[0])));
                                            let _e5976_ = (_e5951_[1] + (_e568_ * (_e5966_ - _e5951_[1])));
                                            let _e5978_ = (_e5951_[2] + (_e588_ * (_e5967_ - _e5951_[2])));
                                            let _e5980_ = (_e5951_[3] + (_e608_ * (_e5968_ - _e5951_[3])));
                                            let _e5987_ = (_e5956_[0] + (_e548_ * (_e5965_ - _e5956_[0])));
                                            let _e5989_ = (_e5956_[1] + (_e568_ * (_e5966_ - _e5956_[1])));
                                            let _e5991_ = (_e5956_[2] + (_e588_ * (_e5967_ - _e5956_[2])));
                                            let _e5993_ = (_e5956_[3] + (_e608_ * (_e5968_ - _e5956_[3])));
                                            let _e6010_ = select(0f, _e458_.x, (bitcast<f32>((bitcast<u32>(_e458_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6012_ = select(0f, _e458_.y, (bitcast<f32>((bitcast<u32>(_e458_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6014_ = select(0f, _e458_.z, (bitcast<f32>((bitcast<u32>(_e458_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6016_ = select(0f, _e458_.w, (bitcast<f32>((bitcast<u32>(_e458_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6017_ = (_e6010_ != _e6010_);
                                            if _e6017_ {
                                                phi_28594_ = true;
                                            } else {
                                                phi_28594_ = (0f >= _e6010_);
                                            }
                                            let _e6020_ = phi_28594_;
                                            let _e6021_ = select(_e6010_, 0f, _e6020_);
                                            let _e6022_ = (_e6012_ != _e6012_);
                                            if _e6022_ {
                                                phi_28609_ = true;
                                            } else {
                                                phi_28609_ = (0f >= _e6012_);
                                            }
                                            let _e6025_ = phi_28609_;
                                            let _e6026_ = select(_e6012_, 0f, _e6025_);
                                            let _e6027_ = (_e6014_ != _e6014_);
                                            if _e6027_ {
                                                phi_28624_ = true;
                                            } else {
                                                phi_28624_ = (0f >= _e6014_);
                                            }
                                            let _e6030_ = phi_28624_;
                                            let _e6031_ = select(_e6014_, 0f, _e6030_);
                                            let _e6032_ = (_e6016_ != _e6016_);
                                            if _e6032_ {
                                                phi_28639_ = true;
                                            } else {
                                                phi_28639_ = (0f >= _e6016_);
                                            }
                                            let _e6035_ = phi_28639_;
                                            let _e6036_ = select(_e6016_, 0f, _e6035_);
                                            if (_e6021_ != _e6021_) {
                                                phi_28654_ = true;
                                            } else {
                                                phi_28654_ = (2.5f <= _e6021_);
                                            }
                                            let _e6040_ = phi_28654_;
                                            let _e6041_ = select(_e6021_, 2.5f, _e6040_);
                                            if (_e6026_ != _e6026_) {
                                                phi_28669_ = true;
                                            } else {
                                                phi_28669_ = (2.5f <= _e6026_);
                                            }
                                            let _e6045_ = phi_28669_;
                                            let _e6046_ = select(_e6026_, 2.5f, _e6045_);
                                            if (_e6031_ != _e6031_) {
                                                phi_28684_ = true;
                                            } else {
                                                phi_28684_ = (2.5f <= _e6031_);
                                            }
                                            let _e6050_ = phi_28684_;
                                            let _e6051_ = select(_e6031_, 2.5f, _e6050_);
                                            if (_e6036_ != _e6036_) {
                                                phi_28699_ = true;
                                            } else {
                                                phi_28699_ = (2.5f <= _e6036_);
                                            }
                                            let _e6055_ = phi_28699_;
                                            let _e6056_ = select(_e6036_, 2.5f, _e6055_);
                                            let _e6070_ = select(0f, _e465_.x, (bitcast<f32>((bitcast<u32>(_e465_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6072_ = select(0f, _e465_.y, (bitcast<f32>((bitcast<u32>(_e465_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6074_ = select(0f, _e465_.z, (bitcast<f32>((bitcast<u32>(_e465_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6076_ = select(0f, _e465_.w, (bitcast<f32>((bitcast<u32>(_e465_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6077_ = (_e6070_ != _e6070_);
                                            if _e6077_ {
                                                phi_28714_ = true;
                                            } else {
                                                phi_28714_ = (0f >= _e6070_);
                                            }
                                            let _e6080_ = phi_28714_;
                                            let _e6081_ = select(_e6070_, 0f, _e6080_);
                                            let _e6082_ = (_e6072_ != _e6072_);
                                            if _e6082_ {
                                                phi_28729_ = true;
                                            } else {
                                                phi_28729_ = (0f >= _e6072_);
                                            }
                                            let _e6085_ = phi_28729_;
                                            let _e6086_ = select(_e6072_, 0f, _e6085_);
                                            let _e6087_ = (_e6074_ != _e6074_);
                                            if _e6087_ {
                                                phi_28744_ = true;
                                            } else {
                                                phi_28744_ = (0f >= _e6074_);
                                            }
                                            let _e6090_ = phi_28744_;
                                            let _e6091_ = select(_e6074_, 0f, _e6090_);
                                            let _e6092_ = (_e6076_ != _e6076_);
                                            if _e6092_ {
                                                phi_28759_ = true;
                                            } else {
                                                phi_28759_ = (0f >= _e6076_);
                                            }
                                            let _e6095_ = phi_28759_;
                                            let _e6096_ = select(_e6076_, 0f, _e6095_);
                                            if (_e6081_ != _e6081_) {
                                                phi_28774_ = true;
                                            } else {
                                                phi_28774_ = (2.5f <= _e6081_);
                                            }
                                            let _e6100_ = phi_28774_;
                                            let _e6101_ = select(_e6081_, 2.5f, _e6100_);
                                            if (_e6086_ != _e6086_) {
                                                phi_28789_ = true;
                                            } else {
                                                phi_28789_ = (2.5f <= _e6086_);
                                            }
                                            let _e6105_ = phi_28789_;
                                            let _e6106_ = select(_e6086_, 2.5f, _e6105_);
                                            if (_e6091_ != _e6091_) {
                                                phi_28804_ = true;
                                            } else {
                                                phi_28804_ = (2.5f <= _e6091_);
                                            }
                                            let _e6110_ = phi_28804_;
                                            let _e6111_ = select(_e6091_, 2.5f, _e6110_);
                                            if (_e6096_ != _e6096_) {
                                                phi_28819_ = true;
                                            } else {
                                                phi_28819_ = (2.5f <= _e6096_);
                                            }
                                            let _e6115_ = phi_28819_;
                                            let _e6116_ = select(_e6096_, 2.5f, _e6115_);
                                            let _e6130_ = select(0f, _e472_.x, (bitcast<f32>((bitcast<u32>(_e472_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6132_ = select(0f, _e472_.y, (bitcast<f32>((bitcast<u32>(_e472_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6134_ = select(0f, _e472_.z, (bitcast<f32>((bitcast<u32>(_e472_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6136_ = select(0f, _e472_.w, (bitcast<f32>((bitcast<u32>(_e472_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6137_ = (_e6130_ != _e6130_);
                                            if _e6137_ {
                                                phi_28834_ = true;
                                            } else {
                                                phi_28834_ = (0f >= _e6130_);
                                            }
                                            let _e6140_ = phi_28834_;
                                            let _e6141_ = select(_e6130_, 0f, _e6140_);
                                            let _e6142_ = (_e6132_ != _e6132_);
                                            if _e6142_ {
                                                phi_28849_ = true;
                                            } else {
                                                phi_28849_ = (0f >= _e6132_);
                                            }
                                            let _e6145_ = phi_28849_;
                                            let _e6146_ = select(_e6132_, 0f, _e6145_);
                                            let _e6147_ = (_e6134_ != _e6134_);
                                            if _e6147_ {
                                                phi_28864_ = true;
                                            } else {
                                                phi_28864_ = (0f >= _e6134_);
                                            }
                                            let _e6150_ = phi_28864_;
                                            let _e6151_ = select(_e6134_, 0f, _e6150_);
                                            let _e6152_ = (_e6136_ != _e6136_);
                                            if _e6152_ {
                                                phi_28879_ = true;
                                            } else {
                                                phi_28879_ = (0f >= _e6136_);
                                            }
                                            let _e6155_ = phi_28879_;
                                            let _e6156_ = select(_e6136_, 0f, _e6155_);
                                            if (_e6141_ != _e6141_) {
                                                phi_28894_ = true;
                                            } else {
                                                phi_28894_ = (1000000000000f <= _e6141_);
                                            }
                                            let _e6160_ = phi_28894_;
                                            if (_e6146_ != _e6146_) {
                                                phi_28909_ = true;
                                            } else {
                                                phi_28909_ = (1000000000000f <= _e6146_);
                                            }
                                            let _e6165_ = phi_28909_;
                                            if (_e6151_ != _e6151_) {
                                                phi_28924_ = true;
                                            } else {
                                                phi_28924_ = (1000000000000f <= _e6151_);
                                            }
                                            let _e6170_ = phi_28924_;
                                            if (_e6156_ != _e6156_) {
                                                phi_28939_ = true;
                                            } else {
                                                phi_28939_ = (1000000000000f <= _e6156_);
                                            }
                                            let _e6175_ = phi_28939_;
                                            let _e6177_ = bitcast<u32>(_e5974_);
                                            let _e6178_ = (_e6177_ & 2139095040u);
                                            if (_e6178_ == 2139095040u) {
                                                phi_18295_ = true;
                                            } else {
                                                phi_18295_ = (_e6178_ == 0u);
                                            }
                                            let _e6182_ = phi_18295_;
                                            let _e6185_ = bitcast<u32>(_e5976_);
                                            let _e6186_ = (_e6185_ & 2139095040u);
                                            if (_e6186_ == 2139095040u) {
                                                phi_18312_ = true;
                                            } else {
                                                phi_18312_ = (_e6186_ == 0u);
                                            }
                                            let _e6190_ = phi_18312_;
                                            let _e6193_ = bitcast<u32>(_e5978_);
                                            let _e6194_ = (_e6193_ & 2139095040u);
                                            if (_e6194_ == 2139095040u) {
                                                phi_18329_ = true;
                                            } else {
                                                phi_18329_ = (_e6194_ == 0u);
                                            }
                                            let _e6198_ = phi_18329_;
                                            let _e6201_ = bitcast<u32>(_e5980_);
                                            let _e6202_ = (_e6201_ & 2139095040u);
                                            if (_e6202_ == 2139095040u) {
                                                phi_18346_ = true;
                                            } else {
                                                phi_18346_ = (_e6202_ == 0u);
                                            }
                                            let _e6206_ = phi_18346_;
                                            let _e6211_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6177_, 0u, _e6182_))) & 2147483647u));
                                            let _e6214_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6185_, 0u, _e6190_))) & 2147483647u));
                                            let _e6217_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6193_, 0u, _e6198_))) & 2147483647u));
                                            let _e6220_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6201_, 0u, _e6206_))) & 2147483647u));
                                            if (_e6211_ != _e6211_) {
                                                phi_28954_ = true;
                                            } else {
                                                phi_28954_ = (1000f <= _e6211_);
                                            }
                                            let _e6224_ = phi_28954_;
                                            let _e6225_ = select(_e6211_, 1000f, _e6224_);
                                            if (_e6214_ != _e6214_) {
                                                phi_28969_ = true;
                                            } else {
                                                phi_28969_ = (1000f <= _e6214_);
                                            }
                                            let _e6229_ = phi_28969_;
                                            let _e6230_ = select(_e6214_, 1000f, _e6229_);
                                            if (_e6217_ != _e6217_) {
                                                phi_28984_ = true;
                                            } else {
                                                phi_28984_ = (1000f <= _e6217_);
                                            }
                                            let _e6234_ = phi_28984_;
                                            let _e6235_ = select(_e6217_, 1000f, _e6234_);
                                            if (_e6220_ != _e6220_) {
                                                phi_28999_ = true;
                                            } else {
                                                phi_28999_ = (1000f <= _e6220_);
                                            }
                                            let _e6239_ = phi_28999_;
                                            let _e6240_ = select(_e6220_, 1000f, _e6239_);
                                            let _e6241_ = (_e6225_ * _e6225_);
                                            let _e6242_ = (_e6230_ * _e6230_);
                                            let _e6243_ = (_e6235_ * _e6235_);
                                            let _e6244_ = (_e6240_ * _e6240_);
                                            if (_e6241_ != _e6241_) {
                                                phi_29014_ = true;
                                            } else {
                                                phi_29014_ = (1000000f <= _e6241_);
                                            }
                                            let _e6248_ = phi_29014_;
                                            if (_e6242_ != _e6242_) {
                                                phi_29029_ = true;
                                            } else {
                                                phi_29029_ = (1000000f <= _e6242_);
                                            }
                                            let _e6253_ = phi_29029_;
                                            if (_e6243_ != _e6243_) {
                                                phi_29044_ = true;
                                            } else {
                                                phi_29044_ = (1000000f <= _e6243_);
                                            }
                                            let _e6258_ = phi_29044_;
                                            if (_e6244_ != _e6244_) {
                                                phi_29059_ = true;
                                            } else {
                                                phi_29059_ = (1000000f <= _e6244_);
                                            }
                                            let _e6263_ = phi_29059_;
                                            let _e6265_ = (select(_e6141_, 1000000000000f, _e6160_) * select(_e6241_, 1000000f, _e6248_));
                                            let _e6266_ = (select(_e6146_, 1000000000000f, _e6165_) * select(_e6242_, 1000000f, _e6253_));
                                            let _e6267_ = (select(_e6151_, 1000000000000f, _e6170_) * select(_e6243_, 1000000f, _e6258_));
                                            let _e6268_ = (select(_e6156_, 1000000000000f, _e6175_) * select(_e6244_, 1000000f, _e6263_));
                                            if (_e6265_ != _e6265_) {
                                                phi_29074_ = true;
                                            } else {
                                                phi_29074_ = (1000000000000f <= _e6265_);
                                            }
                                            let _e6272_ = phi_29074_;
                                            let _e6273_ = select(_e6265_, 1000000000000f, _e6272_);
                                            if (_e6266_ != _e6266_) {
                                                phi_29089_ = true;
                                            } else {
                                                phi_29089_ = (1000000000000f <= _e6266_);
                                            }
                                            let _e6277_ = phi_29089_;
                                            let _e6278_ = select(_e6266_, 1000000000000f, _e6277_);
                                            if (_e6267_ != _e6267_) {
                                                phi_29104_ = true;
                                            } else {
                                                phi_29104_ = (1000000000000f <= _e6267_);
                                            }
                                            let _e6282_ = phi_29104_;
                                            let _e6283_ = select(_e6267_, 1000000000000f, _e6282_);
                                            if (_e6268_ != _e6268_) {
                                                phi_29119_ = true;
                                            } else {
                                                phi_29119_ = (1000000000000f <= _e6268_);
                                            }
                                            let _e6287_ = phi_29119_;
                                            let _e6288_ = select(_e6268_, 1000000000000f, _e6287_);
                                            let _e6289_ = (_e6041_ != _e6041_);
                                            if _e6289_ {
                                                phi_29134_ = true;
                                            } else {
                                                phi_29134_ = (_e6101_ <= _e6041_);
                                            }
                                            let _e6292_ = phi_29134_;
                                            let _e6293_ = select(_e6041_, _e6101_, _e6292_);
                                            let _e6294_ = (_e6046_ != _e6046_);
                                            if _e6294_ {
                                                phi_29149_ = true;
                                            } else {
                                                phi_29149_ = (_e6106_ <= _e6046_);
                                            }
                                            let _e6297_ = phi_29149_;
                                            let _e6298_ = select(_e6046_, _e6106_, _e6297_);
                                            let _e6299_ = (_e6051_ != _e6051_);
                                            if _e6299_ {
                                                phi_29164_ = true;
                                            } else {
                                                phi_29164_ = (_e6111_ <= _e6051_);
                                            }
                                            let _e6302_ = phi_29164_;
                                            let _e6303_ = select(_e6051_, _e6111_, _e6302_);
                                            let _e6304_ = (_e6056_ != _e6056_);
                                            if _e6304_ {
                                                phi_29179_ = true;
                                            } else {
                                                phi_29179_ = (_e6116_ <= _e6056_);
                                            }
                                            let _e6307_ = phi_29179_;
                                            let _e6308_ = select(_e6056_, _e6116_, _e6307_);
                                            if _e6289_ {
                                                phi_29194_ = true;
                                            } else {
                                                phi_29194_ = (_e6101_ >= _e6041_);
                                            }
                                            let _e6311_ = phi_29194_;
                                            let _e6312_ = select(_e6041_, _e6101_, _e6311_);
                                            if _e6294_ {
                                                phi_29209_ = true;
                                            } else {
                                                phi_29209_ = (_e6106_ >= _e6046_);
                                            }
                                            let _e6315_ = phi_29209_;
                                            let _e6316_ = select(_e6046_, _e6106_, _e6315_);
                                            if _e6299_ {
                                                phi_29224_ = true;
                                            } else {
                                                phi_29224_ = (_e6111_ >= _e6051_);
                                            }
                                            let _e6319_ = phi_29224_;
                                            let _e6320_ = select(_e6051_, _e6111_, _e6319_);
                                            if _e6304_ {
                                                phi_29239_ = true;
                                            } else {
                                                phi_29239_ = (_e6116_ >= _e6056_);
                                            }
                                            let _e6323_ = phi_29239_;
                                            let _e6324_ = select(_e6056_, _e6116_, _e6323_);
                                            let _e6333_ = (1f + _e6273_);
                                            let _e6334_ = (1f + _e6278_);
                                            let _e6335_ = (1f + _e6283_);
                                            let _e6336_ = (1f + _e6288_);
                                            let _e6339_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6333_)));
                                            let _e6342_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6334_)));
                                            let _e6345_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6335_)));
                                            let _e6348_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6336_)));
                                            let _e6358_ = (_e6339_ + (_e6339_ * (1f - (_e6333_ * _e6339_))));
                                            let _e6360_ = (_e6342_ + (_e6342_ * (1f - (_e6334_ * _e6342_))));
                                            let _e6362_ = (_e6345_ + (_e6345_ * (1f - (_e6335_ * _e6345_))));
                                            let _e6364_ = (_e6348_ + (_e6348_ * (1f - (_e6336_ * _e6348_))));
                                            let _e6381_ = ((_e6041_ + (_e6101_ * _e6273_)) * (_e6358_ + (_e6358_ * (1f - (_e6333_ * _e6358_)))));
                                            let _e6382_ = ((_e6046_ + (_e6106_ * _e6278_)) * (_e6360_ + (_e6360_ * (1f - (_e6334_ * _e6360_)))));
                                            let _e6383_ = ((_e6051_ + (_e6111_ * _e6283_)) * (_e6362_ + (_e6362_ * (1f - (_e6335_ * _e6362_)))));
                                            let _e6384_ = ((_e6056_ + (_e6116_ * _e6288_)) * (_e6364_ + (_e6364_ * (1f - (_e6336_ * _e6364_)))));
                                            if (_e6381_ != _e6381_) {
                                                phi_29254_ = true;
                                            } else {
                                                phi_29254_ = (_e6293_ >= _e6381_);
                                            }
                                            let _e6388_ = phi_29254_;
                                            let _e6389_ = select(_e6381_, _e6293_, _e6388_);
                                            if (_e6382_ != _e6382_) {
                                                phi_29269_ = true;
                                            } else {
                                                phi_29269_ = (_e6298_ >= _e6382_);
                                            }
                                            let _e6393_ = phi_29269_;
                                            let _e6394_ = select(_e6382_, _e6298_, _e6393_);
                                            if (_e6383_ != _e6383_) {
                                                phi_29284_ = true;
                                            } else {
                                                phi_29284_ = (_e6303_ >= _e6383_);
                                            }
                                            let _e6398_ = phi_29284_;
                                            let _e6399_ = select(_e6383_, _e6303_, _e6398_);
                                            if (_e6384_ != _e6384_) {
                                                phi_29299_ = true;
                                            } else {
                                                phi_29299_ = (_e6308_ >= _e6384_);
                                            }
                                            let _e6403_ = phi_29299_;
                                            let _e6404_ = select(_e6384_, _e6308_, _e6403_);
                                            if (_e6389_ != _e6389_) {
                                                phi_29314_ = true;
                                            } else {
                                                phi_29314_ = (_e6312_ <= _e6389_);
                                            }
                                            let _e6408_ = phi_29314_;
                                            if (_e6394_ != _e6394_) {
                                                phi_29329_ = true;
                                            } else {
                                                phi_29329_ = (_e6316_ <= _e6394_);
                                            }
                                            let _e6413_ = phi_29329_;
                                            if (_e6399_ != _e6399_) {
                                                phi_29344_ = true;
                                            } else {
                                                phi_29344_ = (_e6320_ <= _e6399_);
                                            }
                                            let _e6418_ = phi_29344_;
                                            if (_e6404_ != _e6404_) {
                                                phi_29359_ = true;
                                            } else {
                                                phi_29359_ = (_e6324_ <= _e6404_);
                                            }
                                            let _e6423_ = phi_29359_;
                                            let _e6426_ = select(select(_e6389_, _e6312_, _e6408_), _e6041_, (_e6273_ <= 0f));
                                            let _e6428_ = select(select(_e6394_, _e6316_, _e6413_), _e6046_, (_e6278_ <= 0f));
                                            let _e6430_ = select(select(_e6399_, _e6320_, _e6418_), _e6051_, (_e6283_ <= 0f));
                                            let _e6432_ = select(select(_e6404_, _e6324_, _e6423_), _e6056_, (_e6288_ <= 0f));
                                            if _e6017_ {
                                                phi_29374_ = true;
                                            } else {
                                                phi_29374_ = (0f >= _e6010_);
                                            }
                                            let _e6437_ = phi_29374_;
                                            let _e6438_ = select(_e6010_, 0f, _e6437_);
                                            if _e6022_ {
                                                phi_29389_ = true;
                                            } else {
                                                phi_29389_ = (0f >= _e6012_);
                                            }
                                            let _e6441_ = phi_29389_;
                                            let _e6442_ = select(_e6012_, 0f, _e6441_);
                                            if _e6027_ {
                                                phi_29404_ = true;
                                            } else {
                                                phi_29404_ = (0f >= _e6014_);
                                            }
                                            let _e6445_ = phi_29404_;
                                            let _e6446_ = select(_e6014_, 0f, _e6445_);
                                            if _e6032_ {
                                                phi_29419_ = true;
                                            } else {
                                                phi_29419_ = (0f >= _e6016_);
                                            }
                                            let _e6449_ = phi_29419_;
                                            let _e6450_ = select(_e6016_, 0f, _e6449_);
                                            if (_e6438_ != _e6438_) {
                                                phi_29434_ = true;
                                            } else {
                                                phi_29434_ = (2.5f <= _e6438_);
                                            }
                                            let _e6454_ = phi_29434_;
                                            let _e6455_ = select(_e6438_, 2.5f, _e6454_);
                                            if (_e6442_ != _e6442_) {
                                                phi_29449_ = true;
                                            } else {
                                                phi_29449_ = (2.5f <= _e6442_);
                                            }
                                            let _e6459_ = phi_29449_;
                                            let _e6460_ = select(_e6442_, 2.5f, _e6459_);
                                            if (_e6446_ != _e6446_) {
                                                phi_29464_ = true;
                                            } else {
                                                phi_29464_ = (2.5f <= _e6446_);
                                            }
                                            let _e6464_ = phi_29464_;
                                            let _e6465_ = select(_e6446_, 2.5f, _e6464_);
                                            if (_e6450_ != _e6450_) {
                                                phi_29479_ = true;
                                            } else {
                                                phi_29479_ = (2.5f <= _e6450_);
                                            }
                                            let _e6469_ = phi_29479_;
                                            let _e6470_ = select(_e6450_, 2.5f, _e6469_);
                                            if _e6077_ {
                                                phi_29494_ = true;
                                            } else {
                                                phi_29494_ = (0f >= _e6070_);
                                            }
                                            let _e6473_ = phi_29494_;
                                            let _e6474_ = select(_e6070_, 0f, _e6473_);
                                            if _e6082_ {
                                                phi_29509_ = true;
                                            } else {
                                                phi_29509_ = (0f >= _e6072_);
                                            }
                                            let _e6477_ = phi_29509_;
                                            let _e6478_ = select(_e6072_, 0f, _e6477_);
                                            if _e6087_ {
                                                phi_29524_ = true;
                                            } else {
                                                phi_29524_ = (0f >= _e6074_);
                                            }
                                            let _e6481_ = phi_29524_;
                                            let _e6482_ = select(_e6074_, 0f, _e6481_);
                                            if _e6092_ {
                                                phi_29539_ = true;
                                            } else {
                                                phi_29539_ = (0f >= _e6076_);
                                            }
                                            let _e6485_ = phi_29539_;
                                            let _e6486_ = select(_e6076_, 0f, _e6485_);
                                            if (_e6474_ != _e6474_) {
                                                phi_29554_ = true;
                                            } else {
                                                phi_29554_ = (2.5f <= _e6474_);
                                            }
                                            let _e6490_ = phi_29554_;
                                            let _e6491_ = select(_e6474_, 2.5f, _e6490_);
                                            if (_e6478_ != _e6478_) {
                                                phi_29569_ = true;
                                            } else {
                                                phi_29569_ = (2.5f <= _e6478_);
                                            }
                                            let _e6495_ = phi_29569_;
                                            let _e6496_ = select(_e6478_, 2.5f, _e6495_);
                                            if (_e6482_ != _e6482_) {
                                                phi_29584_ = true;
                                            } else {
                                                phi_29584_ = (2.5f <= _e6482_);
                                            }
                                            let _e6500_ = phi_29584_;
                                            let _e6501_ = select(_e6482_, 2.5f, _e6500_);
                                            if (_e6486_ != _e6486_) {
                                                phi_29599_ = true;
                                            } else {
                                                phi_29599_ = (2.5f <= _e6486_);
                                            }
                                            let _e6505_ = phi_29599_;
                                            let _e6506_ = select(_e6486_, 2.5f, _e6505_);
                                            if _e6137_ {
                                                phi_29614_ = true;
                                            } else {
                                                phi_29614_ = (0f >= _e6130_);
                                            }
                                            let _e6509_ = phi_29614_;
                                            let _e6510_ = select(_e6130_, 0f, _e6509_);
                                            if _e6142_ {
                                                phi_29629_ = true;
                                            } else {
                                                phi_29629_ = (0f >= _e6132_);
                                            }
                                            let _e6513_ = phi_29629_;
                                            let _e6514_ = select(_e6132_, 0f, _e6513_);
                                            if _e6147_ {
                                                phi_29644_ = true;
                                            } else {
                                                phi_29644_ = (0f >= _e6134_);
                                            }
                                            let _e6517_ = phi_29644_;
                                            let _e6518_ = select(_e6134_, 0f, _e6517_);
                                            if _e6152_ {
                                                phi_29659_ = true;
                                            } else {
                                                phi_29659_ = (0f >= _e6136_);
                                            }
                                            let _e6521_ = phi_29659_;
                                            let _e6522_ = select(_e6136_, 0f, _e6521_);
                                            if (_e6510_ != _e6510_) {
                                                phi_29674_ = true;
                                            } else {
                                                phi_29674_ = (1000000000000f <= _e6510_);
                                            }
                                            let _e6526_ = phi_29674_;
                                            if (_e6514_ != _e6514_) {
                                                phi_29689_ = true;
                                            } else {
                                                phi_29689_ = (1000000000000f <= _e6514_);
                                            }
                                            let _e6531_ = phi_29689_;
                                            if (_e6518_ != _e6518_) {
                                                phi_29704_ = true;
                                            } else {
                                                phi_29704_ = (1000000000000f <= _e6518_);
                                            }
                                            let _e6536_ = phi_29704_;
                                            if (_e6522_ != _e6522_) {
                                                phi_29719_ = true;
                                            } else {
                                                phi_29719_ = (1000000000000f <= _e6522_);
                                            }
                                            let _e6541_ = phi_29719_;
                                            let _e6543_ = bitcast<u32>(_e5987_);
                                            let _e6544_ = (_e6543_ & 2139095040u);
                                            if (_e6544_ == 2139095040u) {
                                                phi_18630_ = true;
                                            } else {
                                                phi_18630_ = (_e6544_ == 0u);
                                            }
                                            let _e6548_ = phi_18630_;
                                            let _e6551_ = bitcast<u32>(_e5989_);
                                            let _e6552_ = (_e6551_ & 2139095040u);
                                            if (_e6552_ == 2139095040u) {
                                                phi_18647_ = true;
                                            } else {
                                                phi_18647_ = (_e6552_ == 0u);
                                            }
                                            let _e6556_ = phi_18647_;
                                            let _e6559_ = bitcast<u32>(_e5991_);
                                            let _e6560_ = (_e6559_ & 2139095040u);
                                            if (_e6560_ == 2139095040u) {
                                                phi_18664_ = true;
                                            } else {
                                                phi_18664_ = (_e6560_ == 0u);
                                            }
                                            let _e6564_ = phi_18664_;
                                            let _e6567_ = bitcast<u32>(_e5993_);
                                            let _e6568_ = (_e6567_ & 2139095040u);
                                            if (_e6568_ == 2139095040u) {
                                                phi_18681_ = true;
                                            } else {
                                                phi_18681_ = (_e6568_ == 0u);
                                            }
                                            let _e6572_ = phi_18681_;
                                            let _e6577_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6543_, 0u, _e6548_))) & 2147483647u));
                                            let _e6580_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6551_, 0u, _e6556_))) & 2147483647u));
                                            let _e6583_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6559_, 0u, _e6564_))) & 2147483647u));
                                            let _e6586_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e6567_, 0u, _e6572_))) & 2147483647u));
                                            if (_e6577_ != _e6577_) {
                                                phi_29734_ = true;
                                            } else {
                                                phi_29734_ = (1000f <= _e6577_);
                                            }
                                            let _e6590_ = phi_29734_;
                                            let _e6591_ = select(_e6577_, 1000f, _e6590_);
                                            if (_e6580_ != _e6580_) {
                                                phi_29749_ = true;
                                            } else {
                                                phi_29749_ = (1000f <= _e6580_);
                                            }
                                            let _e6595_ = phi_29749_;
                                            let _e6596_ = select(_e6580_, 1000f, _e6595_);
                                            if (_e6583_ != _e6583_) {
                                                phi_29764_ = true;
                                            } else {
                                                phi_29764_ = (1000f <= _e6583_);
                                            }
                                            let _e6600_ = phi_29764_;
                                            let _e6601_ = select(_e6583_, 1000f, _e6600_);
                                            if (_e6586_ != _e6586_) {
                                                phi_29779_ = true;
                                            } else {
                                                phi_29779_ = (1000f <= _e6586_);
                                            }
                                            let _e6605_ = phi_29779_;
                                            let _e6606_ = select(_e6586_, 1000f, _e6605_);
                                            let _e6607_ = (_e6591_ * _e6591_);
                                            let _e6608_ = (_e6596_ * _e6596_);
                                            let _e6609_ = (_e6601_ * _e6601_);
                                            let _e6610_ = (_e6606_ * _e6606_);
                                            if (_e6607_ != _e6607_) {
                                                phi_29794_ = true;
                                            } else {
                                                phi_29794_ = (1000000f <= _e6607_);
                                            }
                                            let _e6614_ = phi_29794_;
                                            if (_e6608_ != _e6608_) {
                                                phi_29809_ = true;
                                            } else {
                                                phi_29809_ = (1000000f <= _e6608_);
                                            }
                                            let _e6619_ = phi_29809_;
                                            if (_e6609_ != _e6609_) {
                                                phi_29824_ = true;
                                            } else {
                                                phi_29824_ = (1000000f <= _e6609_);
                                            }
                                            let _e6624_ = phi_29824_;
                                            if (_e6610_ != _e6610_) {
                                                phi_29839_ = true;
                                            } else {
                                                phi_29839_ = (1000000f <= _e6610_);
                                            }
                                            let _e6629_ = phi_29839_;
                                            let _e6631_ = (select(_e6510_, 1000000000000f, _e6526_) * select(_e6607_, 1000000f, _e6614_));
                                            let _e6632_ = (select(_e6514_, 1000000000000f, _e6531_) * select(_e6608_, 1000000f, _e6619_));
                                            let _e6633_ = (select(_e6518_, 1000000000000f, _e6536_) * select(_e6609_, 1000000f, _e6624_));
                                            let _e6634_ = (select(_e6522_, 1000000000000f, _e6541_) * select(_e6610_, 1000000f, _e6629_));
                                            if (_e6631_ != _e6631_) {
                                                phi_29854_ = true;
                                            } else {
                                                phi_29854_ = (1000000000000f <= _e6631_);
                                            }
                                            let _e6638_ = phi_29854_;
                                            let _e6639_ = select(_e6631_, 1000000000000f, _e6638_);
                                            if (_e6632_ != _e6632_) {
                                                phi_29869_ = true;
                                            } else {
                                                phi_29869_ = (1000000000000f <= _e6632_);
                                            }
                                            let _e6643_ = phi_29869_;
                                            let _e6644_ = select(_e6632_, 1000000000000f, _e6643_);
                                            if (_e6633_ != _e6633_) {
                                                phi_29884_ = true;
                                            } else {
                                                phi_29884_ = (1000000000000f <= _e6633_);
                                            }
                                            let _e6648_ = phi_29884_;
                                            let _e6649_ = select(_e6633_, 1000000000000f, _e6648_);
                                            if (_e6634_ != _e6634_) {
                                                phi_29899_ = true;
                                            } else {
                                                phi_29899_ = (1000000000000f <= _e6634_);
                                            }
                                            let _e6653_ = phi_29899_;
                                            let _e6654_ = select(_e6634_, 1000000000000f, _e6653_);
                                            let _e6655_ = (_e6455_ != _e6455_);
                                            if _e6655_ {
                                                phi_29914_ = true;
                                            } else {
                                                phi_29914_ = (_e6491_ <= _e6455_);
                                            }
                                            let _e6658_ = phi_29914_;
                                            let _e6659_ = select(_e6455_, _e6491_, _e6658_);
                                            let _e6660_ = (_e6460_ != _e6460_);
                                            if _e6660_ {
                                                phi_29929_ = true;
                                            } else {
                                                phi_29929_ = (_e6496_ <= _e6460_);
                                            }
                                            let _e6663_ = phi_29929_;
                                            let _e6664_ = select(_e6460_, _e6496_, _e6663_);
                                            let _e6665_ = (_e6465_ != _e6465_);
                                            if _e6665_ {
                                                phi_29944_ = true;
                                            } else {
                                                phi_29944_ = (_e6501_ <= _e6465_);
                                            }
                                            let _e6668_ = phi_29944_;
                                            let _e6669_ = select(_e6465_, _e6501_, _e6668_);
                                            let _e6670_ = (_e6470_ != _e6470_);
                                            if _e6670_ {
                                                phi_29959_ = true;
                                            } else {
                                                phi_29959_ = (_e6506_ <= _e6470_);
                                            }
                                            let _e6673_ = phi_29959_;
                                            let _e6674_ = select(_e6470_, _e6506_, _e6673_);
                                            if _e6655_ {
                                                phi_29974_ = true;
                                            } else {
                                                phi_29974_ = (_e6491_ >= _e6455_);
                                            }
                                            let _e6677_ = phi_29974_;
                                            let _e6678_ = select(_e6455_, _e6491_, _e6677_);
                                            if _e6660_ {
                                                phi_29989_ = true;
                                            } else {
                                                phi_29989_ = (_e6496_ >= _e6460_);
                                            }
                                            let _e6681_ = phi_29989_;
                                            let _e6682_ = select(_e6460_, _e6496_, _e6681_);
                                            if _e6665_ {
                                                phi_30004_ = true;
                                            } else {
                                                phi_30004_ = (_e6501_ >= _e6465_);
                                            }
                                            let _e6685_ = phi_30004_;
                                            let _e6686_ = select(_e6465_, _e6501_, _e6685_);
                                            if _e6670_ {
                                                phi_30019_ = true;
                                            } else {
                                                phi_30019_ = (_e6506_ >= _e6470_);
                                            }
                                            let _e6689_ = phi_30019_;
                                            let _e6690_ = select(_e6470_, _e6506_, _e6689_);
                                            let _e6699_ = (1f + _e6639_);
                                            let _e6700_ = (1f + _e6644_);
                                            let _e6701_ = (1f + _e6649_);
                                            let _e6702_ = (1f + _e6654_);
                                            let _e6705_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6699_)));
                                            let _e6708_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6700_)));
                                            let _e6711_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6701_)));
                                            let _e6714_ = bitcast<f32>((2129859011u - bitcast<u32>(_e6702_)));
                                            let _e6724_ = (_e6705_ + (_e6705_ * (1f - (_e6699_ * _e6705_))));
                                            let _e6726_ = (_e6708_ + (_e6708_ * (1f - (_e6700_ * _e6708_))));
                                            let _e6728_ = (_e6711_ + (_e6711_ * (1f - (_e6701_ * _e6711_))));
                                            let _e6730_ = (_e6714_ + (_e6714_ * (1f - (_e6702_ * _e6714_))));
                                            let _e6747_ = ((_e6455_ + (_e6491_ * _e6639_)) * (_e6724_ + (_e6724_ * (1f - (_e6699_ * _e6724_)))));
                                            let _e6748_ = ((_e6460_ + (_e6496_ * _e6644_)) * (_e6726_ + (_e6726_ * (1f - (_e6700_ * _e6726_)))));
                                            let _e6749_ = ((_e6465_ + (_e6501_ * _e6649_)) * (_e6728_ + (_e6728_ * (1f - (_e6701_ * _e6728_)))));
                                            let _e6750_ = ((_e6470_ + (_e6506_ * _e6654_)) * (_e6730_ + (_e6730_ * (1f - (_e6702_ * _e6730_)))));
                                            if (_e6747_ != _e6747_) {
                                                phi_30034_ = true;
                                            } else {
                                                phi_30034_ = (_e6659_ >= _e6747_);
                                            }
                                            let _e6754_ = phi_30034_;
                                            let _e6755_ = select(_e6747_, _e6659_, _e6754_);
                                            if (_e6748_ != _e6748_) {
                                                phi_30049_ = true;
                                            } else {
                                                phi_30049_ = (_e6664_ >= _e6748_);
                                            }
                                            let _e6759_ = phi_30049_;
                                            let _e6760_ = select(_e6748_, _e6664_, _e6759_);
                                            if (_e6749_ != _e6749_) {
                                                phi_30064_ = true;
                                            } else {
                                                phi_30064_ = (_e6669_ >= _e6749_);
                                            }
                                            let _e6764_ = phi_30064_;
                                            let _e6765_ = select(_e6749_, _e6669_, _e6764_);
                                            if (_e6750_ != _e6750_) {
                                                phi_30079_ = true;
                                            } else {
                                                phi_30079_ = (_e6674_ >= _e6750_);
                                            }
                                            let _e6769_ = phi_30079_;
                                            let _e6770_ = select(_e6750_, _e6674_, _e6769_);
                                            if (_e6755_ != _e6755_) {
                                                phi_30094_ = true;
                                            } else {
                                                phi_30094_ = (_e6678_ <= _e6755_);
                                            }
                                            let _e6774_ = phi_30094_;
                                            if (_e6760_ != _e6760_) {
                                                phi_30109_ = true;
                                            } else {
                                                phi_30109_ = (_e6682_ <= _e6760_);
                                            }
                                            let _e6779_ = phi_30109_;
                                            if (_e6765_ != _e6765_) {
                                                phi_30124_ = true;
                                            } else {
                                                phi_30124_ = (_e6686_ <= _e6765_);
                                            }
                                            let _e6784_ = phi_30124_;
                                            if (_e6770_ != _e6770_) {
                                                phi_30139_ = true;
                                            } else {
                                                phi_30139_ = (_e6690_ <= _e6770_);
                                            }
                                            let _e6789_ = phi_30139_;
                                            let _e6792_ = select(select(_e6755_, _e6678_, _e6774_), _e6455_, (_e6639_ <= 0f));
                                            let _e6794_ = select(select(_e6760_, _e6682_, _e6779_), _e6460_, (_e6644_ <= 0f));
                                            let _e6796_ = select(select(_e6765_, _e6686_, _e6784_), _e6465_, (_e6649_ <= 0f));
                                            let _e6798_ = select(select(_e6770_, _e6690_, _e6789_), _e6470_, (_e6654_ <= 0f));
                                            let _e6810_ = (((_e6426_ * _e646_) * _e5937_[0]) + (_e718_ * _e5974_));
                                            let _e6812_ = (((_e6428_ * _e664_) * _e5937_[1]) + (_e736_ * _e5976_));
                                            let _e6814_ = (((_e6430_ * _e682_) * _e5937_[2]) + (_e754_ * _e5978_));
                                            let _e6816_ = (((_e6432_ * _e700_) * _e5937_[3]) + (_e772_ * _e5980_));
                                            let _e6830_ = select(0f, _e6810_, (bitcast<f32>((bitcast<u32>(_e6810_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6832_ = select(0f, _e6812_, (bitcast<f32>((bitcast<u32>(_e6812_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6834_ = select(0f, _e6814_, (bitcast<f32>((bitcast<u32>(_e6814_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6836_ = select(0f, _e6816_, (bitcast<f32>((bitcast<u32>(_e6816_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e6830_ != _e6830_) {
                                                phi_30154_ = true;
                                            } else {
                                                phi_30154_ = (-4096f >= _e6830_);
                                            }
                                            let _e6840_ = phi_30154_;
                                            let _e6841_ = select(_e6830_, -4096f, _e6840_);
                                            if (_e6832_ != _e6832_) {
                                                phi_30169_ = true;
                                            } else {
                                                phi_30169_ = (-4096f >= _e6832_);
                                            }
                                            let _e6845_ = phi_30169_;
                                            let _e6846_ = select(_e6832_, -4096f, _e6845_);
                                            if (_e6834_ != _e6834_) {
                                                phi_30184_ = true;
                                            } else {
                                                phi_30184_ = (-4096f >= _e6834_);
                                            }
                                            let _e6850_ = phi_30184_;
                                            let _e6851_ = select(_e6834_, -4096f, _e6850_);
                                            if (_e6836_ != _e6836_) {
                                                phi_30199_ = true;
                                            } else {
                                                phi_30199_ = (-4096f >= _e6836_);
                                            }
                                            let _e6855_ = phi_30199_;
                                            let _e6856_ = select(_e6836_, -4096f, _e6855_);
                                            if (_e6841_ != _e6841_) {
                                                phi_30214_ = true;
                                            } else {
                                                phi_30214_ = (4096f <= _e6841_);
                                            }
                                            let _e6860_ = phi_30214_;
                                            let _e6861_ = select(_e6841_, 4096f, _e6860_);
                                            if (_e6846_ != _e6846_) {
                                                phi_30229_ = true;
                                            } else {
                                                phi_30229_ = (4096f <= _e6846_);
                                            }
                                            let _e6865_ = phi_30229_;
                                            let _e6866_ = select(_e6846_, 4096f, _e6865_);
                                            if (_e6851_ != _e6851_) {
                                                phi_30244_ = true;
                                            } else {
                                                phi_30244_ = (4096f <= _e6851_);
                                            }
                                            let _e6870_ = phi_30244_;
                                            let _e6871_ = select(_e6851_, 4096f, _e6870_);
                                            if (_e6856_ != _e6856_) {
                                                phi_30259_ = true;
                                            } else {
                                                phi_30259_ = (4096f <= _e6856_);
                                            }
                                            let _e6875_ = phi_30259_;
                                            let _e6876_ = select(_e6856_, 4096f, _e6875_);
                                            let _e6888_ = (((_e6792_ * _e646_) * _e5944_[0]) + (_e718_ * _e5987_));
                                            let _e6890_ = (((_e6794_ * _e664_) * _e5944_[1]) + (_e736_ * _e5989_));
                                            let _e6892_ = (((_e6796_ * _e682_) * _e5944_[2]) + (_e754_ * _e5991_));
                                            let _e6894_ = (((_e6798_ * _e700_) * _e5944_[3]) + (_e772_ * _e5993_));
                                            let _e6908_ = select(0f, _e6888_, (bitcast<f32>((bitcast<u32>(_e6888_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6910_ = select(0f, _e6890_, (bitcast<f32>((bitcast<u32>(_e6890_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6912_ = select(0f, _e6892_, (bitcast<f32>((bitcast<u32>(_e6892_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e6914_ = select(0f, _e6894_, (bitcast<f32>((bitcast<u32>(_e6894_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e6908_ != _e6908_) {
                                                phi_30274_ = true;
                                            } else {
                                                phi_30274_ = (-4096f >= _e6908_);
                                            }
                                            let _e6918_ = phi_30274_;
                                            let _e6919_ = select(_e6908_, -4096f, _e6918_);
                                            if (_e6910_ != _e6910_) {
                                                phi_30289_ = true;
                                            } else {
                                                phi_30289_ = (-4096f >= _e6910_);
                                            }
                                            let _e6923_ = phi_30289_;
                                            let _e6924_ = select(_e6910_, -4096f, _e6923_);
                                            if (_e6912_ != _e6912_) {
                                                phi_30304_ = true;
                                            } else {
                                                phi_30304_ = (-4096f >= _e6912_);
                                            }
                                            let _e6928_ = phi_30304_;
                                            let _e6929_ = select(_e6912_, -4096f, _e6928_);
                                            if (_e6914_ != _e6914_) {
                                                phi_30319_ = true;
                                            } else {
                                                phi_30319_ = (-4096f >= _e6914_);
                                            }
                                            let _e6933_ = phi_30319_;
                                            let _e6934_ = select(_e6914_, -4096f, _e6933_);
                                            if (_e6919_ != _e6919_) {
                                                phi_30334_ = true;
                                            } else {
                                                phi_30334_ = (4096f <= _e6919_);
                                            }
                                            let _e6938_ = phi_30334_;
                                            let _e6939_ = select(_e6919_, 4096f, _e6938_);
                                            if (_e6924_ != _e6924_) {
                                                phi_30349_ = true;
                                            } else {
                                                phi_30349_ = (4096f <= _e6924_);
                                            }
                                            let _e6943_ = phi_30349_;
                                            let _e6944_ = select(_e6924_, 4096f, _e6943_);
                                            if (_e6929_ != _e6929_) {
                                                phi_30364_ = true;
                                            } else {
                                                phi_30364_ = (4096f <= _e6929_);
                                            }
                                            let _e6948_ = phi_30364_;
                                            let _e6949_ = select(_e6929_, 4096f, _e6948_);
                                            if (_e6934_ != _e6934_) {
                                                phi_30379_ = true;
                                            } else {
                                                phi_30379_ = (4096f <= _e6934_);
                                            }
                                            let _e6953_ = phi_30379_;
                                            let _e6954_ = select(_e6934_, 4096f, _e6953_);
                                            local_17_ = array<f32, 4>(_e6861_, _e6866_, _e6871_, _e6876_);
                                            local_18_ = array<f32, 4>(_e6939_, _e6944_, _e6949_, _e6954_);
                                            phi_18979_ = type_15_(0u, 4u);
                                            loop {
                                                let _e6960_ = phi_18979_;
                                                if (_e6960_.member < _e6960_.member_1_) {
                                                    phi_18980_ = type_15_((_e6960_.member + 1u), _e6960_.member_1_);
                                                    phi_19003_ = type_15_(1u, _e6960_.member);
                                                } else {
                                                    phi_18980_ = _e6960_;
                                                    phi_19003_ = type_15_(0u, u32());
                                                }
                                                let _e6973_ = phi_18980_;
                                                let _e6975_ = phi_19003_;
                                                let _e6979_ = (bitcast<i32>(_e6975_.member) != 0i);
                                                if _e6979_ {
                                                    let _e6980_ = (_e6975_.member_1_ < 4u);
                                                    if _e6980_ {
                                                    } else {
                                                        phi_37004_ = true;
                                                        break;
                                                    }
                                                    let _e6982_ = local_17_[_e6975_.member_1_];
                                                    let _e6984_ = ((_e6975_.member_1_ * 128u) + _e197_);
                                                    if (_e6984_ < 1536u) {
                                                    } else {
                                                        phi_37004_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6984_] = _e6982_;
                                                    if _e6980_ {
                                                    } else {
                                                        phi_37004_ = true;
                                                        break;
                                                    }
                                                    let _e6988_ = local_18_[_e6975_.member_1_];
                                                    let _e6991_ = (((4u + _e6975_.member_1_) * 128u) + _e197_);
                                                    if (_e6991_ < 1536u) {
                                                    } else {
                                                        phi_37004_ = true;
                                                        break;
                                                    }
                                                    scan_scratch[_e6991_] = _e6988_;
                                                }
                                                phi_18979_ = _e6973_;
                                                phi_37004_ = _e5936_;
                                                if !(_e6979_) {
                                                    break;
                                                }
                                            }
                                            let _e6996_ = phi_37004_;
                                            phi_38285_ = _e6996_;
                                            if _e6996_ {
                                                break;
                                            }
                                            phi_37309_ = _e6996_;
                                            phi_19042_ = type_32_(array<f32, 4>(_e6939_, _e6944_, _e6949_, _e6954_));
                                            phi_19043_ = type_32_(array<f32, 4>(_e6861_, _e6866_, _e6871_, _e6876_));
                                            phi_19044_ = type_32_(array<f32, 4>(_e6792_, _e6794_, _e6796_, _e6798_));
                                            phi_19045_ = type_32_(array<f32, 4>(_e6426_, _e6428_, _e6430_, _e6432_));
                                            phi_19046_ = type_32_(array<f32, 4>(_e5944_[0], _e5944_[1], _e5944_[2], _e5944_[3]));
                                            phi_19047_ = type_32_(array<f32, 4>(_e5937_[0], _e5937_[1], _e5937_[2], _e5937_[3]));
                                            phi_19048_ = type_32_(array<f32, 4>(_e5987_, _e5989_, _e5991_, _e5993_));
                                            phi_19049_ = type_32_(array<f32, 4>(_e5974_, _e5976_, _e5978_, _e5980_));
                                        } else {
                                            phi_37309_ = _e5881_;
                                            phi_19042_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19043_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19044_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19045_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19046_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19047_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19048_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                            phi_19049_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        }
                                        let _e6998_ = phi_37309_;
                                        let _e7000_ = phi_19042_;
                                        let _e7002_ = phi_19043_;
                                        let _e7004_ = phi_19044_;
                                        let _e7006_ = phi_19045_;
                                        let _e7008_ = phi_19046_;
                                        let _e7010_ = phi_19047_;
                                        let _e7012_ = phi_19048_;
                                        let _e7014_ = phi_19049_;
                                        phi_37308_ = _e6998_;
                                        phi_19050_ = _e7000_;
                                        phi_19051_ = _e7002_;
                                        phi_19052_ = _e7004_;
                                        phi_19053_ = _e7006_;
                                        phi_19054_ = _e7008_;
                                        phi_19055_ = _e7010_;
                                        phi_19056_ = _e7012_;
                                        phi_19057_ = _e7014_;
                                    } else {
                                        phi_37308_ = _e5881_;
                                        phi_19050_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19051_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19052_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19053_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19054_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19055_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19056_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                        phi_19057_ = type_32_(array<f32, 4>(0f, 0f, 0f, 0f));
                                    }
                                    let _e7016_ = phi_37308_;
                                    let _e7018_ = phi_19050_;
                                    let _e7020_ = phi_19051_;
                                    let _e7022_ = phi_19052_;
                                    let _e7024_ = phi_19053_;
                                    let _e7026_ = phi_19054_;
                                    let _e7028_ = phi_19055_;
                                    let _e7030_ = phi_19056_;
                                    let _e7032_ = phi_19057_;
                                    workgroupBarrier();
                                    if _e5882_ {
                                        if _e2406_ {
                                            local_19_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            local_20_ = array<f32, 4>(0f, 0f, 0f, 0f);
                                            if (_e197_ == 0u) {
                                                phi_19071_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e7074_ = phi_19071_;
                                                    if (_e7074_.member < _e7074_.member_1_) {
                                                        phi_19072_ = type_15_((_e7074_.member + 1u), _e7074_.member_1_);
                                                        phi_19095_ = type_15_(1u, _e7074_.member);
                                                    } else {
                                                        phi_19072_ = _e7074_;
                                                        phi_19095_ = type_15_(0u, u32());
                                                    }
                                                    let _e7087_ = phi_19072_;
                                                    let _e7089_ = phi_19095_;
                                                    let _e7093_ = (bitcast<i32>(_e7089_.member) != 0i);
                                                    if _e7093_ {
                                                        let _e7094_ = (24u + _e7089_.member_1_);
                                                        if (_e7094_ < 256u) {
                                                        } else {
                                                            phi_37310_ = true;
                                                            break;
                                                        }
                                                        let _e7097_ = aux_scratch[_e7094_];
                                                        let _e7098_ = (_e7089_.member_1_ < 4u);
                                                        if _e7098_ {
                                                        } else {
                                                            phi_37310_ = true;
                                                            break;
                                                        }
                                                        local_19_[_e7089_.member_1_] = _e7097_;
                                                        let _e7100_ = (28u + _e7089_.member_1_);
                                                        if (_e7100_ < 256u) {
                                                        } else {
                                                            phi_37310_ = true;
                                                            break;
                                                        }
                                                        let _e7103_ = aux_scratch[_e7100_];
                                                        if _e7098_ {
                                                        } else {
                                                            phi_37310_ = true;
                                                            break;
                                                        }
                                                        local_20_[_e7089_.member_1_] = _e7103_;
                                                    }
                                                    phi_19071_ = _e7087_;
                                                    phi_37310_ = _e7016_;
                                                    if !(_e7093_) {
                                                        break;
                                                    }
                                                }
                                                let _e7107_ = phi_37310_;
                                                phi_38285_ = _e7107_;
                                                if _e7107_ {
                                                    break;
                                                }
                                                phi_38276_ = _e7107_;
                                            } else {
                                                let _e7034_ = (_e197_ - 1u);
                                                phi_19134_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e7036_ = phi_19134_;
                                                    if (_e7036_.member < _e7036_.member_1_) {
                                                        phi_19135_ = type_15_((_e7036_.member + 1u), _e7036_.member_1_);
                                                        phi_19158_ = type_15_(1u, _e7036_.member);
                                                    } else {
                                                        phi_19135_ = _e7036_;
                                                        phi_19158_ = type_15_(0u, u32());
                                                    }
                                                    let _e7049_ = phi_19135_;
                                                    let _e7051_ = phi_19158_;
                                                    let _e7055_ = (bitcast<i32>(_e7051_.member) != 0i);
                                                    if _e7055_ {
                                                        let _e7057_ = ((_e7051_.member_1_ * 128u) + _e7034_);
                                                        if (_e7057_ < 1536u) {
                                                        } else {
                                                            phi_37304_ = true;
                                                            break;
                                                        }
                                                        let _e7060_ = scan_scratch[_e7057_];
                                                        let _e7061_ = (_e7051_.member_1_ < 4u);
                                                        if _e7061_ {
                                                        } else {
                                                            phi_37304_ = true;
                                                            break;
                                                        }
                                                        local_19_[_e7051_.member_1_] = _e7060_;
                                                        let _e7065_ = (((4u + _e7051_.member_1_) * 128u) + _e7034_);
                                                        if (_e7065_ < 1536u) {
                                                        } else {
                                                            phi_37304_ = true;
                                                            break;
                                                        }
                                                        let _e7068_ = scan_scratch[_e7065_];
                                                        if _e7061_ {
                                                        } else {
                                                            phi_37304_ = true;
                                                            break;
                                                        }
                                                        local_20_[_e7051_.member_1_] = _e7068_;
                                                    }
                                                    phi_19134_ = _e7049_;
                                                    phi_37304_ = _e7016_;
                                                    if !(_e7055_) {
                                                        break;
                                                    }
                                                }
                                                let _e7072_ = phi_37304_;
                                                phi_38285_ = _e7072_;
                                                if _e7072_ {
                                                    break;
                                                }
                                                phi_38276_ = _e7072_;
                                            }
                                            let _e7109_ = phi_38276_;
                                            let _e7110_ = local_19_;
                                            let _e7128_ = select(0f, _e862_, (bitcast<f32>((bitcast<u32>(_e862_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7130_ = select(0f, _e880_, (bitcast<f32>((bitcast<u32>(_e880_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7132_ = select(0f, _e898_, (bitcast<f32>((bitcast<u32>(_e898_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7134_ = select(0f, _e916_, (bitcast<f32>((bitcast<u32>(_e916_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7135_ = (_e7128_ != _e7128_);
                                            if _e7135_ {
                                                phi_30394_ = true;
                                            } else {
                                                phi_30394_ = (0f >= _e7128_);
                                            }
                                            let _e7138_ = phi_30394_;
                                            let _e7139_ = select(_e7128_, 0f, _e7138_);
                                            let _e7140_ = (_e7130_ != _e7130_);
                                            if _e7140_ {
                                                phi_30409_ = true;
                                            } else {
                                                phi_30409_ = (0f >= _e7130_);
                                            }
                                            let _e7143_ = phi_30409_;
                                            let _e7144_ = select(_e7130_, 0f, _e7143_);
                                            let _e7145_ = (_e7132_ != _e7132_);
                                            if _e7145_ {
                                                phi_30424_ = true;
                                            } else {
                                                phi_30424_ = (0f >= _e7132_);
                                            }
                                            let _e7148_ = phi_30424_;
                                            let _e7149_ = select(_e7132_, 0f, _e7148_);
                                            let _e7150_ = (_e7134_ != _e7134_);
                                            if _e7150_ {
                                                phi_30439_ = true;
                                            } else {
                                                phi_30439_ = (0f >= _e7134_);
                                            }
                                            let _e7153_ = phi_30439_;
                                            let _e7154_ = select(_e7134_, 0f, _e7153_);
                                            if (_e7139_ != _e7139_) {
                                                phi_30454_ = true;
                                            } else {
                                                phi_30454_ = (1f <= _e7139_);
                                            }
                                            let _e7158_ = phi_30454_;
                                            let _e7159_ = select(_e7139_, 1f, _e7158_);
                                            if (_e7144_ != _e7144_) {
                                                phi_30469_ = true;
                                            } else {
                                                phi_30469_ = (1f <= _e7144_);
                                            }
                                            let _e7163_ = phi_30469_;
                                            let _e7164_ = select(_e7144_, 1f, _e7163_);
                                            if (_e7149_ != _e7149_) {
                                                phi_30484_ = true;
                                            } else {
                                                phi_30484_ = (1f <= _e7149_);
                                            }
                                            let _e7168_ = phi_30484_;
                                            let _e7169_ = select(_e7149_, 1f, _e7168_);
                                            if (_e7154_ != _e7154_) {
                                                phi_30499_ = true;
                                            } else {
                                                phi_30499_ = (1f <= _e7154_);
                                            }
                                            let _e7173_ = phi_30499_;
                                            let _e7174_ = select(_e7154_, 1f, _e7173_);
                                            let _e7188_ = select(0f, _e7110_[0], (bitcast<f32>((bitcast<u32>(_e7110_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7190_ = select(0f, _e7110_[1], (bitcast<f32>((bitcast<u32>(_e7110_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7192_ = select(0f, _e7110_[2], (bitcast<f32>((bitcast<u32>(_e7110_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7194_ = select(0f, _e7110_[3], (bitcast<f32>((bitcast<u32>(_e7110_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7195_ = (_e7188_ != _e7188_);
                                            if _e7195_ {
                                                phi_30514_ = true;
                                            } else {
                                                phi_30514_ = (-4096f >= _e7188_);
                                            }
                                            let _e7198_ = phi_30514_;
                                            let _e7199_ = select(_e7188_, -4096f, _e7198_);
                                            let _e7200_ = (_e7190_ != _e7190_);
                                            if _e7200_ {
                                                phi_30529_ = true;
                                            } else {
                                                phi_30529_ = (-4096f >= _e7190_);
                                            }
                                            let _e7203_ = phi_30529_;
                                            let _e7204_ = select(_e7190_, -4096f, _e7203_);
                                            let _e7205_ = (_e7192_ != _e7192_);
                                            if _e7205_ {
                                                phi_30544_ = true;
                                            } else {
                                                phi_30544_ = (-4096f >= _e7192_);
                                            }
                                            let _e7208_ = phi_30544_;
                                            let _e7209_ = select(_e7192_, -4096f, _e7208_);
                                            let _e7210_ = (_e7194_ != _e7194_);
                                            if _e7210_ {
                                                phi_30559_ = true;
                                            } else {
                                                phi_30559_ = (-4096f >= _e7194_);
                                            }
                                            let _e7213_ = phi_30559_;
                                            let _e7214_ = select(_e7194_, -4096f, _e7213_);
                                            if (_e7199_ != _e7199_) {
                                                phi_30574_ = true;
                                            } else {
                                                phi_30574_ = (4096f <= _e7199_);
                                            }
                                            let _e7218_ = phi_30574_;
                                            let _e7219_ = select(_e7199_, 4096f, _e7218_);
                                            if (_e7204_ != _e7204_) {
                                                phi_30589_ = true;
                                            } else {
                                                phi_30589_ = (4096f <= _e7204_);
                                            }
                                            let _e7223_ = phi_30589_;
                                            let _e7224_ = select(_e7204_, 4096f, _e7223_);
                                            if (_e7209_ != _e7209_) {
                                                phi_30604_ = true;
                                            } else {
                                                phi_30604_ = (4096f <= _e7209_);
                                            }
                                            let _e7228_ = phi_30604_;
                                            let _e7229_ = select(_e7209_, 4096f, _e7228_);
                                            if (_e7214_ != _e7214_) {
                                                phi_30619_ = true;
                                            } else {
                                                phi_30619_ = (4096f <= _e7214_);
                                            }
                                            let _e7233_ = phi_30619_;
                                            let _e7234_ = select(_e7214_, 4096f, _e7233_);
                                            let _e7256_ = select(0f, _e7020_.member[0], (bitcast<f32>((bitcast<u32>(_e7020_.member[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7258_ = select(0f, _e7020_.member[1], (bitcast<f32>((bitcast<u32>(_e7020_.member[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7260_ = select(0f, _e7020_.member[2], (bitcast<f32>((bitcast<u32>(_e7020_.member[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e7262_ = select(0f, _e7020_.member[3], (bitcast<f32>((bitcast<u32>(_e7020_.member[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e7256_ != _e7256_) {
                                                phi_30634_ = true;
                                            } else {
                                                phi_30634_ = (-4096f >= _e7256_);
                                            }
                                            let _e7266_ = phi_30634_;
                                            let _e7267_ = select(_e7256_, -4096f, _e7266_);
                                            if (_e7258_ != _e7258_) {
                                                phi_30649_ = true;
                                            } else {
                                                phi_30649_ = (-4096f >= _e7258_);
                                            }
                                            let _e7271_ = phi_30649_;
                                            let _e7272_ = select(_e7258_, -4096f, _e7271_);
                                            if (_e7260_ != _e7260_) {
                                                phi_30664_ = true;
                                            } else {
                                                phi_30664_ = (-4096f >= _e7260_);
                                            }
                                            let _e7276_ = phi_30664_;
                                            let _e7277_ = select(_e7260_, -4096f, _e7276_);
                                            if (_e7262_ != _e7262_) {
                                                phi_30679_ = true;
                                            } else {
                                                phi_30679_ = (-4096f >= _e7262_);
                                            }
                                            let _e7281_ = phi_30679_;
                                            let _e7282_ = select(_e7262_, -4096f, _e7281_);
                                            if (_e7267_ != _e7267_) {
                                                phi_30694_ = true;
                                            } else {
                                                phi_30694_ = (4096f <= _e7267_);
                                            }
                                            let _e7286_ = phi_30694_;
                                            let _e7287_ = select(_e7267_, 4096f, _e7286_);
                                            if (_e7272_ != _e7272_) {
                                                phi_30709_ = true;
                                            } else {
                                                phi_30709_ = (4096f <= _e7272_);
                                            }
                                            let _e7291_ = phi_30709_;
                                            let _e7292_ = select(_e7272_, 4096f, _e7291_);
                                            if (_e7277_ != _e7277_) {
                                                phi_30724_ = true;
                                            } else {
                                                phi_30724_ = (4096f <= _e7277_);
                                            }
                                            let _e7296_ = phi_30724_;
                                            let _e7297_ = select(_e7277_, 4096f, _e7296_);
                                            if (_e7282_ != _e7282_) {
                                                phi_30739_ = true;
                                            } else {
                                                phi_30739_ = (4096f <= _e7282_);
                                            }
                                            let _e7301_ = phi_30739_;
                                            let _e7302_ = select(_e7282_, 4096f, _e7301_);
                                            let _e7303_ = array<f32, 4>(_e7287_, _e7292_, _e7297_, _e7302_);
                                            let _e7304_ = (_e7287_ * _e7287_);
                                            let _e7305_ = (_e7292_ * _e7292_);
                                            let _e7306_ = (_e7297_ * _e7297_);
                                            let _e7307_ = (_e7302_ * _e7302_);
                                            let _e7308_ = (_e7219_ * _e7219_);
                                            let _e7309_ = (_e7224_ * _e7224_);
                                            let _e7310_ = (_e7229_ * _e7229_);
                                            let _e7311_ = (_e7234_ * _e7234_);
                                            let _e7312_ = (1f + _e7304_);
                                            let _e7313_ = (1f + _e7305_);
                                            let _e7314_ = (1f + _e7306_);
                                            let _e7315_ = (1f + _e7307_);
                                            let _e7316_ = (1f + _e7308_);
                                            let _e7317_ = (1f + _e7309_);
                                            let _e7318_ = (1f + _e7310_);
                                            let _e7319_ = (1f + _e7311_);
                                            if (_e7312_ != _e7312_) {
                                                phi_30754_ = true;
                                            } else {
                                                phi_30754_ = (0.000000000001f >= _e7312_);
                                            }
                                            let _e7323_ = phi_30754_;
                                            let _e7324_ = select(_e7312_, 0.000000000001f, _e7323_);
                                            if (_e7313_ != _e7313_) {
                                                phi_30769_ = true;
                                            } else {
                                                phi_30769_ = (0.000000000001f >= _e7313_);
                                            }
                                            let _e7328_ = phi_30769_;
                                            let _e7329_ = select(_e7313_, 0.000000000001f, _e7328_);
                                            if (_e7314_ != _e7314_) {
                                                phi_30784_ = true;
                                            } else {
                                                phi_30784_ = (0.000000000001f >= _e7314_);
                                            }
                                            let _e7333_ = phi_30784_;
                                            let _e7334_ = select(_e7314_, 0.000000000001f, _e7333_);
                                            if (_e7315_ != _e7315_) {
                                                phi_30799_ = true;
                                            } else {
                                                phi_30799_ = (0.000000000001f >= _e7315_);
                                            }
                                            let _e7338_ = phi_30799_;
                                            let _e7339_ = select(_e7315_, 0.000000000001f, _e7338_);
                                            let _e7340_ = (_e7324_ * 0.5f);
                                            let _e7341_ = (_e7329_ * 0.5f);
                                            let _e7342_ = (_e7334_ * 0.5f);
                                            let _e7343_ = (_e7339_ * 0.5f);
                                            let _e7348_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7324_) >> bitcast<u32>(1i))));
                                            let _e7353_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7329_) >> bitcast<u32>(1i))));
                                            let _e7358_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7334_) >> bitcast<u32>(1i))));
                                            let _e7363_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7339_) >> bitcast<u32>(1i))));
                                            let _e7376_ = (_e7348_ * (1.5f - (_e7340_ * (_e7348_ * _e7348_))));
                                            let _e7377_ = (_e7353_ * (1.5f - (_e7341_ * (_e7353_ * _e7353_))));
                                            let _e7378_ = (_e7358_ * (1.5f - (_e7342_ * (_e7358_ * _e7358_))));
                                            let _e7379_ = (_e7363_ * (1.5f - (_e7343_ * (_e7363_ * _e7363_))));
                                            let _e7392_ = (_e7376_ * (1.5f - (_e7340_ * (_e7376_ * _e7376_))));
                                            let _e7393_ = (_e7377_ * (1.5f - (_e7341_ * (_e7377_ * _e7377_))));
                                            let _e7394_ = (_e7378_ * (1.5f - (_e7342_ * (_e7378_ * _e7378_))));
                                            let _e7395_ = (_e7379_ * (1.5f - (_e7343_ * (_e7379_ * _e7379_))));
                                            let _e7408_ = (_e7392_ * (1.5f - (_e7340_ * (_e7392_ * _e7392_))));
                                            let _e7409_ = (_e7393_ * (1.5f - (_e7341_ * (_e7393_ * _e7393_))));
                                            let _e7410_ = (_e7394_ * (1.5f - (_e7342_ * (_e7394_ * _e7394_))));
                                            let _e7411_ = (_e7395_ * (1.5f - (_e7343_ * (_e7395_ * _e7395_))));
                                            if (_e7316_ != _e7316_) {
                                                phi_30814_ = true;
                                            } else {
                                                phi_30814_ = (0.000000000001f >= _e7316_);
                                            }
                                            let _e7415_ = phi_30814_;
                                            let _e7416_ = select(_e7316_, 0.000000000001f, _e7415_);
                                            if (_e7317_ != _e7317_) {
                                                phi_30829_ = true;
                                            } else {
                                                phi_30829_ = (0.000000000001f >= _e7317_);
                                            }
                                            let _e7420_ = phi_30829_;
                                            let _e7421_ = select(_e7317_, 0.000000000001f, _e7420_);
                                            if (_e7318_ != _e7318_) {
                                                phi_30844_ = true;
                                            } else {
                                                phi_30844_ = (0.000000000001f >= _e7318_);
                                            }
                                            let _e7425_ = phi_30844_;
                                            let _e7426_ = select(_e7318_, 0.000000000001f, _e7425_);
                                            if (_e7319_ != _e7319_) {
                                                phi_30859_ = true;
                                            } else {
                                                phi_30859_ = (0.000000000001f >= _e7319_);
                                            }
                                            let _e7430_ = phi_30859_;
                                            let _e7431_ = select(_e7319_, 0.000000000001f, _e7430_);
                                            let _e7432_ = (_e7416_ * 0.5f);
                                            let _e7433_ = (_e7421_ * 0.5f);
                                            let _e7434_ = (_e7426_ * 0.5f);
                                            let _e7435_ = (_e7431_ * 0.5f);
                                            let _e7440_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7416_) >> bitcast<u32>(1i))));
                                            let _e7445_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7421_) >> bitcast<u32>(1i))));
                                            let _e7450_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7426_) >> bitcast<u32>(1i))));
                                            let _e7455_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e7431_) >> bitcast<u32>(1i))));
                                            let _e7468_ = (_e7440_ * (1.5f - (_e7432_ * (_e7440_ * _e7440_))));
                                            let _e7469_ = (_e7445_ * (1.5f - (_e7433_ * (_e7445_ * _e7445_))));
                                            let _e7470_ = (_e7450_ * (1.5f - (_e7434_ * (_e7450_ * _e7450_))));
                                            let _e7471_ = (_e7455_ * (1.5f - (_e7435_ * (_e7455_ * _e7455_))));
                                            let _e7484_ = (_e7468_ * (1.5f - (_e7432_ * (_e7468_ * _e7468_))));
                                            let _e7485_ = (_e7469_ * (1.5f - (_e7433_ * (_e7469_ * _e7469_))));
                                            let _e7486_ = (_e7470_ * (1.5f - (_e7434_ * (_e7470_ * _e7470_))));
                                            let _e7487_ = (_e7471_ * (1.5f - (_e7435_ * (_e7471_ * _e7471_))));
                                            let _e7500_ = (_e7484_ * (1.5f - (_e7432_ * (_e7484_ * _e7484_))));
                                            let _e7501_ = (_e7485_ * (1.5f - (_e7433_ * (_e7485_ * _e7485_))));
                                            let _e7502_ = (_e7486_ * (1.5f - (_e7434_ * (_e7486_ * _e7486_))));
                                            let _e7503_ = (_e7487_ * (1.5f - (_e7435_ * (_e7487_ * _e7487_))));
                                            let _e7504_ = (_e7312_ * _e7408_);
                                            let _e7505_ = (_e7313_ * _e7409_);
                                            let _e7506_ = (_e7314_ * _e7410_);
                                            let _e7507_ = (_e7315_ * _e7411_);
                                            let _e7508_ = (_e7316_ * _e7500_);
                                            let _e7509_ = (_e7317_ * _e7501_);
                                            let _e7510_ = (_e7318_ * _e7502_);
                                            let _e7511_ = (_e7319_ * _e7503_);
                                            let _e7512_ = (_e7504_ + 1f);
                                            let _e7513_ = (_e7505_ + 1f);
                                            let _e7514_ = (_e7506_ + 1f);
                                            let _e7515_ = (_e7507_ + 1f);
                                            let _e7518_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7512_)));
                                            let _e7521_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7513_)));
                                            let _e7524_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7514_)));
                                            let _e7527_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7515_)));
                                            let _e7537_ = (_e7518_ + (_e7518_ * (1f - (_e7512_ * _e7518_))));
                                            let _e7539_ = (_e7521_ + (_e7521_ * (1f - (_e7513_ * _e7521_))));
                                            let _e7541_ = (_e7524_ + (_e7524_ * (1f - (_e7514_ * _e7524_))));
                                            let _e7543_ = (_e7527_ + (_e7527_ * (1f - (_e7515_ * _e7527_))));
                                            let _e7553_ = (_e7537_ + (_e7537_ * (1f - (_e7512_ * _e7537_))));
                                            let _e7555_ = (_e7539_ + (_e7539_ * (1f - (_e7513_ * _e7539_))));
                                            let _e7557_ = (_e7541_ + (_e7541_ * (1f - (_e7514_ * _e7541_))));
                                            let _e7559_ = (_e7543_ + (_e7543_ * (1f - (_e7515_ * _e7543_))));
                                            let _e7576_ = (_e7304_ * (_e7553_ + (_e7553_ * (1f - (_e7512_ * _e7553_)))));
                                            let _e7577_ = (_e7305_ * (_e7555_ + (_e7555_ * (1f - (_e7513_ * _e7555_)))));
                                            let _e7578_ = (_e7306_ * (_e7557_ + (_e7557_ * (1f - (_e7514_ * _e7557_)))));
                                            let _e7579_ = (_e7307_ * (_e7559_ + (_e7559_ * (1f - (_e7515_ * _e7559_)))));
                                            let _e7580_ = (_e7508_ + 1f);
                                            let _e7581_ = (_e7509_ + 1f);
                                            let _e7582_ = (_e7510_ + 1f);
                                            let _e7583_ = (_e7511_ + 1f);
                                            let _e7586_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7580_)));
                                            let _e7589_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7581_)));
                                            let _e7592_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7582_)));
                                            let _e7595_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7583_)));
                                            let _e7605_ = (_e7586_ + (_e7586_ * (1f - (_e7580_ * _e7586_))));
                                            let _e7607_ = (_e7589_ + (_e7589_ * (1f - (_e7581_ * _e7589_))));
                                            let _e7609_ = (_e7592_ + (_e7592_ * (1f - (_e7582_ * _e7592_))));
                                            let _e7611_ = (_e7595_ + (_e7595_ * (1f - (_e7583_ * _e7595_))));
                                            let _e7621_ = (_e7605_ + (_e7605_ * (1f - (_e7580_ * _e7605_))));
                                            let _e7623_ = (_e7607_ + (_e7607_ * (1f - (_e7581_ * _e7607_))));
                                            let _e7625_ = (_e7609_ + (_e7609_ * (1f - (_e7582_ * _e7609_))));
                                            let _e7627_ = (_e7611_ + (_e7611_ * (1f - (_e7583_ * _e7611_))));
                                            let _e7644_ = (_e7308_ * (_e7621_ + (_e7621_ * (1f - (_e7580_ * _e7621_)))));
                                            let _e7645_ = (_e7309_ * (_e7623_ + (_e7623_ * (1f - (_e7581_ * _e7623_)))));
                                            let _e7646_ = (_e7310_ * (_e7625_ + (_e7625_ * (1f - (_e7582_ * _e7625_)))));
                                            let _e7647_ = (_e7311_ * (_e7627_ + (_e7627_ * (1f - (_e7583_ * _e7627_)))));
                                            let _e7648_ = (_e7504_ + _e7508_);
                                            let _e7649_ = (_e7505_ + _e7509_);
                                            let _e7650_ = (_e7506_ + _e7510_);
                                            let _e7651_ = (_e7507_ + _e7511_);
                                            let _e7654_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7648_)));
                                            let _e7657_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7649_)));
                                            let _e7660_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7650_)));
                                            let _e7663_ = bitcast<f32>((2129859011u - bitcast<u32>(_e7651_)));
                                            let _e7673_ = (_e7654_ + (_e7654_ * (1f - (_e7648_ * _e7654_))));
                                            let _e7675_ = (_e7657_ + (_e7657_ * (1f - (_e7649_ * _e7657_))));
                                            let _e7677_ = (_e7660_ + (_e7660_ * (1f - (_e7650_ * _e7660_))));
                                            let _e7679_ = (_e7663_ + (_e7663_ * (1f - (_e7651_ * _e7663_))));
                                            let _e7689_ = (_e7673_ + (_e7673_ * (1f - (_e7648_ * _e7673_))));
                                            let _e7691_ = (_e7675_ + (_e7675_ * (1f - (_e7649_ * _e7675_))));
                                            let _e7693_ = (_e7677_ + (_e7677_ * (1f - (_e7650_ * _e7677_))));
                                            let _e7695_ = (_e7679_ + (_e7679_ * (1f - (_e7651_ * _e7679_))));
                                            let _e7705_ = (_e7689_ + (_e7689_ * (1f - (_e7648_ * _e7689_))));
                                            let _e7707_ = (_e7691_ + (_e7691_ * (1f - (_e7649_ * _e7691_))));
                                            let _e7709_ = (_e7693_ + (_e7693_ * (1f - (_e7650_ * _e7693_))));
                                            let _e7711_ = (_e7695_ + (_e7695_ * (1f - (_e7651_ * _e7695_))));
                                            let _e7732_ = (((_e7287_ + _e7219_) * _e7705_) * (1f + (0.5f * (_e7408_ * _e7500_))));
                                            let _e7733_ = (((_e7292_ + _e7224_) * _e7707_) * (1f + (0.5f * (_e7409_ * _e7501_))));
                                            let _e7734_ = (((_e7297_ + _e7229_) * _e7709_) * (1f + (0.5f * (_e7410_ * _e7502_))));
                                            let _e7735_ = (((_e7302_ + _e7234_) * _e7711_) * (1f + (0.5f * (_e7411_ * _e7503_))));
                                            let _e7736_ = (_e7576_ + _e7644_);
                                            let _e7737_ = (_e7577_ + _e7645_);
                                            let _e7738_ = (_e7578_ + _e7646_);
                                            let _e7739_ = (_e7579_ + _e7647_);
                                            let _e7780_ = ((((((_e7736_ * (2f + _e7736_)) + ((_e7576_ * _e7644_) * (1f + _e7736_))) + (_e7287_ * _e7219_)) * _e7705_) * _e7408_) * _e7500_);
                                            let _e7781_ = ((((((_e7737_ * (2f + _e7737_)) + ((_e7577_ * _e7645_) * (1f + _e7737_))) + (_e7292_ * _e7224_)) * _e7707_) * _e7409_) * _e7501_);
                                            let _e7782_ = ((((((_e7738_ * (2f + _e7738_)) + ((_e7578_ * _e7646_) * (1f + _e7738_))) + (_e7297_ * _e7229_)) * _e7709_) * _e7410_) * _e7502_);
                                            let _e7783_ = ((((((_e7739_ * (2f + _e7739_)) + ((_e7579_ * _e7647_) * (1f + _e7739_))) + (_e7302_ * _e7234_)) * _e7711_) * _e7411_) * _e7503_);
                                            let _e7786_ = bitcast<f32>((bitcast<u32>(_e7219_) & 2147483647u));
                                            let _e7789_ = bitcast<f32>((bitcast<u32>(_e7224_) & 2147483647u));
                                            let _e7792_ = bitcast<f32>((bitcast<u32>(_e7229_) & 2147483647u));
                                            let _e7795_ = bitcast<f32>((bitcast<u32>(_e7234_) & 2147483647u));
                                            let _e7798_ = bitcast<f32>((bitcast<u32>(_e7287_) & 2147483647u));
                                            let _e7801_ = bitcast<f32>((bitcast<u32>(_e7292_) & 2147483647u));
                                            let _e7804_ = bitcast<f32>((bitcast<u32>(_e7297_) & 2147483647u));
                                            let _e7807_ = bitcast<f32>((bitcast<u32>(_e7302_) & 2147483647u));
                                            if (_e7786_ != _e7786_) {
                                                phi_30874_ = true;
                                            } else {
                                                phi_30874_ = (_e7798_ >= _e7786_);
                                            }
                                            let _e7811_ = phi_30874_;
                                            if (_e7789_ != _e7789_) {
                                                phi_30889_ = true;
                                            } else {
                                                phi_30889_ = (_e7801_ >= _e7789_);
                                            }
                                            let _e7816_ = phi_30889_;
                                            if (_e7792_ != _e7792_) {
                                                phi_30904_ = true;
                                            } else {
                                                phi_30904_ = (_e7804_ >= _e7792_);
                                            }
                                            let _e7821_ = phi_30904_;
                                            if (_e7795_ != _e7795_) {
                                                phi_30919_ = true;
                                            } else {
                                                phi_30919_ = (_e7807_ >= _e7795_);
                                            }
                                            let _e7826_ = phi_30919_;
                                            if (_e7732_ != _e7732_) {
                                                phi_30934_ = true;
                                            } else {
                                                phi_30934_ = (-1f >= _e7732_);
                                            }
                                            let _e7831_ = phi_30934_;
                                            let _e7832_ = select(_e7732_, -1f, _e7831_);
                                            if (_e7733_ != _e7733_) {
                                                phi_30949_ = true;
                                            } else {
                                                phi_30949_ = (-1f >= _e7733_);
                                            }
                                            let _e7836_ = phi_30949_;
                                            let _e7837_ = select(_e7733_, -1f, _e7836_);
                                            if (_e7734_ != _e7734_) {
                                                phi_30964_ = true;
                                            } else {
                                                phi_30964_ = (-1f >= _e7734_);
                                            }
                                            let _e7841_ = phi_30964_;
                                            let _e7842_ = select(_e7734_, -1f, _e7841_);
                                            if (_e7735_ != _e7735_) {
                                                phi_30979_ = true;
                                            } else {
                                                phi_30979_ = (-1f >= _e7735_);
                                            }
                                            let _e7846_ = phi_30979_;
                                            let _e7847_ = select(_e7735_, -1f, _e7846_);
                                            if (_e7832_ != _e7832_) {
                                                phi_30994_ = true;
                                            } else {
                                                phi_30994_ = (1f <= _e7832_);
                                            }
                                            let _e7851_ = phi_30994_;
                                            if (_e7837_ != _e7837_) {
                                                phi_31009_ = true;
                                            } else {
                                                phi_31009_ = (1f <= _e7837_);
                                            }
                                            let _e7856_ = phi_31009_;
                                            if (_e7842_ != _e7842_) {
                                                phi_31024_ = true;
                                            } else {
                                                phi_31024_ = (1f <= _e7842_);
                                            }
                                            let _e7861_ = phi_31024_;
                                            if (_e7847_ != _e7847_) {
                                                phi_31039_ = true;
                                            } else {
                                                phi_31039_ = (1f <= _e7847_);
                                            }
                                            let _e7866_ = phi_31039_;
                                            let _e7868_ = (select(_e7786_, _e7798_, _e7811_) <= 0f);
                                            let _e7869_ = select(select(_e7832_, 1f, _e7851_), 0f, _e7868_);
                                            let _e7870_ = (select(_e7789_, _e7801_, _e7816_) <= 0f);
                                            let _e7871_ = select(select(_e7837_, 1f, _e7856_), 0f, _e7870_);
                                            let _e7872_ = (select(_e7792_, _e7804_, _e7821_) <= 0f);
                                            let _e7873_ = select(select(_e7842_, 1f, _e7861_), 0f, _e7872_);
                                            let _e7874_ = (select(_e7795_, _e7807_, _e7826_) <= 0f);
                                            let _e7875_ = select(select(_e7847_, 1f, _e7866_), 0f, _e7874_);
                                            if (_e7780_ != _e7780_) {
                                                phi_31054_ = true;
                                            } else {
                                                phi_31054_ = (0f >= _e7780_);
                                            }
                                            let _e7880_ = phi_31054_;
                                            let _e7881_ = select(_e7780_, 0f, _e7880_);
                                            if (_e7781_ != _e7781_) {
                                                phi_31069_ = true;
                                            } else {
                                                phi_31069_ = (0f >= _e7781_);
                                            }
                                            let _e7885_ = phi_31069_;
                                            let _e7886_ = select(_e7781_, 0f, _e7885_);
                                            if (_e7782_ != _e7782_) {
                                                phi_31084_ = true;
                                            } else {
                                                phi_31084_ = (0f >= _e7782_);
                                            }
                                            let _e7890_ = phi_31084_;
                                            let _e7891_ = select(_e7782_, 0f, _e7890_);
                                            if (_e7783_ != _e7783_) {
                                                phi_31099_ = true;
                                            } else {
                                                phi_31099_ = (0f >= _e7783_);
                                            }
                                            let _e7895_ = phi_31099_;
                                            let _e7896_ = select(_e7783_, 0f, _e7895_);
                                            if (_e7881_ != _e7881_) {
                                                phi_31114_ = true;
                                            } else {
                                                phi_31114_ = (1f <= _e7881_);
                                            }
                                            let _e7900_ = phi_31114_;
                                            if (_e7886_ != _e7886_) {
                                                phi_31129_ = true;
                                            } else {
                                                phi_31129_ = (1f <= _e7886_);
                                            }
                                            let _e7905_ = phi_31129_;
                                            if (_e7891_ != _e7891_) {
                                                phi_31144_ = true;
                                            } else {
                                                phi_31144_ = (1f <= _e7891_);
                                            }
                                            let _e7910_ = phi_31144_;
                                            if (_e7896_ != _e7896_) {
                                                phi_31159_ = true;
                                            } else {
                                                phi_31159_ = (1f <= _e7896_);
                                            }
                                            let _e7915_ = phi_31159_;
                                            let _e7921_ = array<f32, 4>(select(select(_e7881_, 1f, _e7900_), 0f, _e7868_), select(select(_e7886_, 1f, _e7905_), 0f, _e7870_), select(select(_e7891_, 1f, _e7910_), 0f, _e7872_), select(select(_e7896_, 1f, _e7915_), 0f, _e7874_));
                                            if (_e7159_ <= 0f) {
                                                if (_e7164_ <= 0f) {
                                                    let _e7924_ = (_e7169_ <= 0f);
                                                    if _e7924_ {
                                                        phi_19912_ = (_e7174_ <= 0f);
                                                    } else {
                                                        phi_19912_ = bool();
                                                    }
                                                    let _e7927_ = phi_19912_;
                                                    phi_19914_ = _e7927_;
                                                    phi_19915_ = select(true, false, _e7924_);
                                                } else {
                                                    phi_19914_ = bool();
                                                    phi_19915_ = true;
                                                }
                                                let _e7930_ = phi_19914_;
                                                let _e7932_ = phi_19915_;
                                                phi_19916_ = _e7930_;
                                                phi_19917_ = _e7932_;
                                            } else {
                                                phi_19916_ = bool();
                                                phi_19917_ = true;
                                            }
                                            let _e7934_ = phi_19916_;
                                            let _e7936_ = phi_19917_;
                                            if select(_e7934_, false, _e7936_) {
                                                phi_20605_ = type_33_(type_32_(array<f32, 4>(_e7869_, _e7871_, _e7873_, _e7875_)), type_32_(_e7921_), type_32_(_e7303_));
                                            } else {
                                                if _e7195_ {
                                                    phi_31174_ = true;
                                                } else {
                                                    phi_31174_ = (-4096f >= _e7188_);
                                                }
                                                let _e7940_ = phi_31174_;
                                                let _e7941_ = select(_e7188_, -4096f, _e7940_);
                                                if _e7200_ {
                                                    phi_31189_ = true;
                                                } else {
                                                    phi_31189_ = (-4096f >= _e7190_);
                                                }
                                                let _e7944_ = phi_31189_;
                                                let _e7945_ = select(_e7190_, -4096f, _e7944_);
                                                if _e7205_ {
                                                    phi_31204_ = true;
                                                } else {
                                                    phi_31204_ = (-4096f >= _e7192_);
                                                }
                                                let _e7948_ = phi_31204_;
                                                let _e7949_ = select(_e7192_, -4096f, _e7948_);
                                                if _e7210_ {
                                                    phi_31219_ = true;
                                                } else {
                                                    phi_31219_ = (-4096f >= _e7194_);
                                                }
                                                let _e7952_ = phi_31219_;
                                                let _e7953_ = select(_e7194_, -4096f, _e7952_);
                                                if (_e7941_ != _e7941_) {
                                                    phi_31234_ = true;
                                                } else {
                                                    phi_31234_ = (4096f <= _e7941_);
                                                }
                                                let _e7957_ = phi_31234_;
                                                let _e7958_ = select(_e7941_, 4096f, _e7957_);
                                                if (_e7945_ != _e7945_) {
                                                    phi_31249_ = true;
                                                } else {
                                                    phi_31249_ = (4096f <= _e7945_);
                                                }
                                                let _e7962_ = phi_31249_;
                                                let _e7963_ = select(_e7945_, 4096f, _e7962_);
                                                if (_e7949_ != _e7949_) {
                                                    phi_31264_ = true;
                                                } else {
                                                    phi_31264_ = (4096f <= _e7949_);
                                                }
                                                let _e7967_ = phi_31264_;
                                                let _e7968_ = select(_e7949_, 4096f, _e7967_);
                                                if (_e7953_ != _e7953_) {
                                                    phi_31279_ = true;
                                                } else {
                                                    phi_31279_ = (4096f <= _e7953_);
                                                }
                                                let _e7972_ = phi_31279_;
                                                let _e7973_ = select(_e7953_, 4096f, _e7972_);
                                                let _e7987_ = select(0f, _e7958_, (bitcast<f32>((bitcast<u32>(_e7958_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                let _e7989_ = select(0f, _e7963_, (bitcast<f32>((bitcast<u32>(_e7963_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                let _e7991_ = select(0f, _e7968_, (bitcast<f32>((bitcast<u32>(_e7968_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                let _e7993_ = select(0f, _e7973_, (bitcast<f32>((bitcast<u32>(_e7973_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                if (_e7987_ != _e7987_) {
                                                    phi_31294_ = true;
                                                } else {
                                                    phi_31294_ = (-1000000000000000000f >= _e7987_);
                                                }
                                                let _e7997_ = phi_31294_;
                                                let _e7998_ = select(_e7987_, -1000000000000000000f, _e7997_);
                                                if (_e7989_ != _e7989_) {
                                                    phi_31309_ = true;
                                                } else {
                                                    phi_31309_ = (-1000000000000000000f >= _e7989_);
                                                }
                                                let _e8002_ = phi_31309_;
                                                let _e8003_ = select(_e7989_, -1000000000000000000f, _e8002_);
                                                if (_e7991_ != _e7991_) {
                                                    phi_31324_ = true;
                                                } else {
                                                    phi_31324_ = (-1000000000000000000f >= _e7991_);
                                                }
                                                let _e8007_ = phi_31324_;
                                                let _e8008_ = select(_e7991_, -1000000000000000000f, _e8007_);
                                                if (_e7993_ != _e7993_) {
                                                    phi_31339_ = true;
                                                } else {
                                                    phi_31339_ = (-1000000000000000000f >= _e7993_);
                                                }
                                                let _e8012_ = phi_31339_;
                                                let _e8013_ = select(_e7993_, -1000000000000000000f, _e8012_);
                                                if (_e7998_ != _e7998_) {
                                                    phi_31354_ = true;
                                                } else {
                                                    phi_31354_ = (1000000000000000000f <= _e7998_);
                                                }
                                                let _e8017_ = phi_31354_;
                                                let _e8018_ = select(_e7998_, 1000000000000000000f, _e8017_);
                                                if (_e8003_ != _e8003_) {
                                                    phi_31369_ = true;
                                                } else {
                                                    phi_31369_ = (1000000000000000000f <= _e8003_);
                                                }
                                                let _e8022_ = phi_31369_;
                                                let _e8023_ = select(_e8003_, 1000000000000000000f, _e8022_);
                                                if (_e8008_ != _e8008_) {
                                                    phi_31384_ = true;
                                                } else {
                                                    phi_31384_ = (1000000000000000000f <= _e8008_);
                                                }
                                                let _e8027_ = phi_31384_;
                                                let _e8028_ = select(_e8008_, 1000000000000000000f, _e8027_);
                                                if (_e8013_ != _e8013_) {
                                                    phi_31399_ = true;
                                                } else {
                                                    phi_31399_ = (1000000000000000000f <= _e8013_);
                                                }
                                                let _e8032_ = phi_31399_;
                                                let _e8033_ = select(_e8013_, 1000000000000000000f, _e8032_);
                                                let _e8035_ = select(0f, _e7287_, (_e7798_ <= 340282350000000000000000000000000000000f));
                                                let _e8037_ = select(0f, _e7292_, (_e7801_ <= 340282350000000000000000000000000000000f));
                                                let _e8039_ = select(0f, _e7297_, (_e7804_ <= 340282350000000000000000000000000000000f));
                                                let _e8041_ = select(0f, _e7302_, (_e7807_ <= 340282350000000000000000000000000000000f));
                                                if (_e8035_ != _e8035_) {
                                                    phi_31414_ = true;
                                                } else {
                                                    phi_31414_ = (-1000000000000000000f >= _e8035_);
                                                }
                                                let _e8045_ = phi_31414_;
                                                let _e8046_ = select(_e8035_, -1000000000000000000f, _e8045_);
                                                if (_e8037_ != _e8037_) {
                                                    phi_31429_ = true;
                                                } else {
                                                    phi_31429_ = (-1000000000000000000f >= _e8037_);
                                                }
                                                let _e8050_ = phi_31429_;
                                                let _e8051_ = select(_e8037_, -1000000000000000000f, _e8050_);
                                                if (_e8039_ != _e8039_) {
                                                    phi_31444_ = true;
                                                } else {
                                                    phi_31444_ = (-1000000000000000000f >= _e8039_);
                                                }
                                                let _e8055_ = phi_31444_;
                                                let _e8056_ = select(_e8039_, -1000000000000000000f, _e8055_);
                                                if (_e8041_ != _e8041_) {
                                                    phi_31459_ = true;
                                                } else {
                                                    phi_31459_ = (-1000000000000000000f >= _e8041_);
                                                }
                                                let _e8060_ = phi_31459_;
                                                let _e8061_ = select(_e8041_, -1000000000000000000f, _e8060_);
                                                if (_e8046_ != _e8046_) {
                                                    phi_31474_ = true;
                                                } else {
                                                    phi_31474_ = (1000000000000000000f <= _e8046_);
                                                }
                                                let _e8065_ = phi_31474_;
                                                let _e8066_ = select(_e8046_, 1000000000000000000f, _e8065_);
                                                if (_e8051_ != _e8051_) {
                                                    phi_31489_ = true;
                                                } else {
                                                    phi_31489_ = (1000000000000000000f <= _e8051_);
                                                }
                                                let _e8070_ = phi_31489_;
                                                let _e8071_ = select(_e8051_, 1000000000000000000f, _e8070_);
                                                if (_e8056_ != _e8056_) {
                                                    phi_31504_ = true;
                                                } else {
                                                    phi_31504_ = (1000000000000000000f <= _e8056_);
                                                }
                                                let _e8075_ = phi_31504_;
                                                let _e8076_ = select(_e8056_, 1000000000000000000f, _e8075_);
                                                if (_e8061_ != _e8061_) {
                                                    phi_31519_ = true;
                                                } else {
                                                    phi_31519_ = (1000000000000000000f <= _e8061_);
                                                }
                                                let _e8080_ = phi_31519_;
                                                let _e8081_ = select(_e8061_, 1000000000000000000f, _e8080_);
                                                let _e8082_ = (_e8018_ != _e8018_);
                                                if _e8082_ {
                                                    phi_31534_ = true;
                                                } else {
                                                    phi_31534_ = (_e8066_ <= _e8018_);
                                                }
                                                let _e8085_ = phi_31534_;
                                                let _e8086_ = select(_e8018_, _e8066_, _e8085_);
                                                let _e8087_ = (_e8023_ != _e8023_);
                                                if _e8087_ {
                                                    phi_31549_ = true;
                                                } else {
                                                    phi_31549_ = (_e8071_ <= _e8023_);
                                                }
                                                let _e8090_ = phi_31549_;
                                                let _e8091_ = select(_e8023_, _e8071_, _e8090_);
                                                let _e8092_ = (_e8028_ != _e8028_);
                                                if _e8092_ {
                                                    phi_31564_ = true;
                                                } else {
                                                    phi_31564_ = (_e8076_ <= _e8028_);
                                                }
                                                let _e8095_ = phi_31564_;
                                                let _e8096_ = select(_e8028_, _e8076_, _e8095_);
                                                let _e8097_ = (_e8033_ != _e8033_);
                                                if _e8097_ {
                                                    phi_31579_ = true;
                                                } else {
                                                    phi_31579_ = (_e8081_ <= _e8033_);
                                                }
                                                let _e8100_ = phi_31579_;
                                                let _e8101_ = select(_e8033_, _e8081_, _e8100_);
                                                if _e8082_ {
                                                    phi_31594_ = true;
                                                } else {
                                                    phi_31594_ = (_e8066_ >= _e8018_);
                                                }
                                                let _e8104_ = phi_31594_;
                                                let _e8105_ = select(_e8018_, _e8066_, _e8104_);
                                                if _e8087_ {
                                                    phi_31609_ = true;
                                                } else {
                                                    phi_31609_ = (_e8071_ >= _e8023_);
                                                }
                                                let _e8108_ = phi_31609_;
                                                let _e8109_ = select(_e8023_, _e8071_, _e8108_);
                                                if _e8092_ {
                                                    phi_31624_ = true;
                                                } else {
                                                    phi_31624_ = (_e8076_ >= _e8028_);
                                                }
                                                let _e8112_ = phi_31624_;
                                                let _e8113_ = select(_e8028_, _e8076_, _e8112_);
                                                if _e8097_ {
                                                    phi_31639_ = true;
                                                } else {
                                                    phi_31639_ = (_e8081_ >= _e8033_);
                                                }
                                                let _e8116_ = phi_31639_;
                                                let _e8117_ = select(_e8033_, _e8081_, _e8116_);
                                                if (_e8086_ != _e8086_) {
                                                    phi_31654_ = true;
                                                } else {
                                                    phi_31654_ = (-1f >= _e8086_);
                                                }
                                                let _e8121_ = phi_31654_;
                                                let _e8122_ = select(_e8086_, -1f, _e8121_);
                                                if (_e8091_ != _e8091_) {
                                                    phi_31669_ = true;
                                                } else {
                                                    phi_31669_ = (-1f >= _e8091_);
                                                }
                                                let _e8126_ = phi_31669_;
                                                let _e8127_ = select(_e8091_, -1f, _e8126_);
                                                if (_e8096_ != _e8096_) {
                                                    phi_31684_ = true;
                                                } else {
                                                    phi_31684_ = (-1f >= _e8096_);
                                                }
                                                let _e8131_ = phi_31684_;
                                                let _e8132_ = select(_e8096_, -1f, _e8131_);
                                                if (_e8101_ != _e8101_) {
                                                    phi_31699_ = true;
                                                } else {
                                                    phi_31699_ = (-1f >= _e8101_);
                                                }
                                                let _e8136_ = phi_31699_;
                                                let _e8137_ = select(_e8101_, -1f, _e8136_);
                                                if (_e8122_ != _e8122_) {
                                                    phi_31714_ = true;
                                                } else {
                                                    phi_31714_ = (1f <= _e8122_);
                                                }
                                                let _e8141_ = phi_31714_;
                                                let _e8142_ = select(_e8122_, 1f, _e8141_);
                                                if (_e8127_ != _e8127_) {
                                                    phi_31729_ = true;
                                                } else {
                                                    phi_31729_ = (1f <= _e8127_);
                                                }
                                                let _e8146_ = phi_31729_;
                                                let _e8147_ = select(_e8127_, 1f, _e8146_);
                                                if (_e8132_ != _e8132_) {
                                                    phi_31744_ = true;
                                                } else {
                                                    phi_31744_ = (1f <= _e8132_);
                                                }
                                                let _e8151_ = phi_31744_;
                                                let _e8152_ = select(_e8132_, 1f, _e8151_);
                                                if (_e8137_ != _e8137_) {
                                                    phi_31759_ = true;
                                                } else {
                                                    phi_31759_ = (1f <= _e8137_);
                                                }
                                                let _e8156_ = phi_31759_;
                                                let _e8157_ = select(_e8137_, 1f, _e8156_);
                                                if (_e8105_ != _e8105_) {
                                                    phi_31774_ = true;
                                                } else {
                                                    phi_31774_ = (-1f >= _e8105_);
                                                }
                                                let _e8161_ = phi_31774_;
                                                let _e8162_ = select(_e8105_, -1f, _e8161_);
                                                if (_e8109_ != _e8109_) {
                                                    phi_31789_ = true;
                                                } else {
                                                    phi_31789_ = (-1f >= _e8109_);
                                                }
                                                let _e8166_ = phi_31789_;
                                                let _e8167_ = select(_e8109_, -1f, _e8166_);
                                                if (_e8113_ != _e8113_) {
                                                    phi_31804_ = true;
                                                } else {
                                                    phi_31804_ = (-1f >= _e8113_);
                                                }
                                                let _e8171_ = phi_31804_;
                                                let _e8172_ = select(_e8113_, -1f, _e8171_);
                                                if (_e8117_ != _e8117_) {
                                                    phi_31819_ = true;
                                                } else {
                                                    phi_31819_ = (-1f >= _e8117_);
                                                }
                                                let _e8176_ = phi_31819_;
                                                let _e8177_ = select(_e8117_, -1f, _e8176_);
                                                if (_e8162_ != _e8162_) {
                                                    phi_31834_ = true;
                                                } else {
                                                    phi_31834_ = (1f <= _e8162_);
                                                }
                                                let _e8181_ = phi_31834_;
                                                let _e8182_ = select(_e8162_, 1f, _e8181_);
                                                if (_e8167_ != _e8167_) {
                                                    phi_31849_ = true;
                                                } else {
                                                    phi_31849_ = (1f <= _e8167_);
                                                }
                                                let _e8186_ = phi_31849_;
                                                let _e8187_ = select(_e8167_, 1f, _e8186_);
                                                if (_e8172_ != _e8172_) {
                                                    phi_31864_ = true;
                                                } else {
                                                    phi_31864_ = (1f <= _e8172_);
                                                }
                                                let _e8191_ = phi_31864_;
                                                let _e8192_ = select(_e8172_, 1f, _e8191_);
                                                if (_e8177_ != _e8177_) {
                                                    phi_31879_ = true;
                                                } else {
                                                    phi_31879_ = (1f <= _e8177_);
                                                }
                                                let _e8196_ = phi_31879_;
                                                let _e8197_ = select(_e8177_, 1f, _e8196_);
                                                let _e8202_ = (0.5f * (_e8182_ + _e8142_));
                                                let _e8203_ = (0.5f * (_e8187_ + _e8147_));
                                                let _e8204_ = (0.5f * (_e8192_ + _e8152_));
                                                let _e8205_ = (0.5f * (_e8197_ + _e8157_));
                                                let _e8206_ = (_e8182_ - _e8142_);
                                                let _e8207_ = (_e8187_ - _e8147_);
                                                let _e8208_ = (_e8192_ - _e8152_);
                                                let _e8209_ = (_e8197_ - _e8157_);
                                                let _e8210_ = (0.5f * _e8206_);
                                                let _e8211_ = (0.5f * _e8207_);
                                                let _e8212_ = (0.5f * _e8208_);
                                                let _e8213_ = (0.5f * _e8209_);
                                                let _e8214_ = (_e8202_ * _e8202_);
                                                let _e8215_ = (_e8203_ * _e8203_);
                                                let _e8216_ = (_e8204_ * _e8204_);
                                                let _e8217_ = (_e8205_ * _e8205_);
                                                let _e8218_ = (_e8210_ * _e8210_);
                                                let _e8219_ = (_e8211_ * _e8211_);
                                                let _e8220_ = (_e8212_ * _e8212_);
                                                let _e8221_ = (_e8213_ * _e8213_);
                                                if (_e8206_ != _e8206_) {
                                                    phi_31894_ = true;
                                                } else {
                                                    phi_31894_ = (0f >= _e8206_);
                                                }
                                                let _e8225_ = phi_31894_;
                                                if (_e8207_ != _e8207_) {
                                                    phi_31909_ = true;
                                                } else {
                                                    phi_31909_ = (0f >= _e8207_);
                                                }
                                                let _e8230_ = phi_31909_;
                                                if (_e8208_ != _e8208_) {
                                                    phi_31924_ = true;
                                                } else {
                                                    phi_31924_ = (0f >= _e8208_);
                                                }
                                                let _e8235_ = phi_31924_;
                                                if (_e8209_ != _e8209_) {
                                                    phi_31939_ = true;
                                                } else {
                                                    phi_31939_ = (0f >= _e8209_);
                                                }
                                                let _e8240_ = phi_31939_;
                                                let _e8242_ = (_e8105_ - _e8086_);
                                                let _e8243_ = (_e8109_ - _e8091_);
                                                let _e8244_ = (_e8113_ - _e8096_);
                                                let _e8245_ = (_e8117_ - _e8101_);
                                                let _e8248_ = bitcast<f32>((bitcast<u32>(_e8018_) & 2147483647u));
                                                let _e8251_ = bitcast<f32>((bitcast<u32>(_e8023_) & 2147483647u));
                                                let _e8254_ = bitcast<f32>((bitcast<u32>(_e8028_) & 2147483647u));
                                                let _e8257_ = bitcast<f32>((bitcast<u32>(_e8033_) & 2147483647u));
                                                let _e8260_ = bitcast<f32>((bitcast<u32>(_e8066_) & 2147483647u));
                                                let _e8263_ = bitcast<f32>((bitcast<u32>(_e8071_) & 2147483647u));
                                                let _e8266_ = bitcast<f32>((bitcast<u32>(_e8076_) & 2147483647u));
                                                let _e8269_ = bitcast<f32>((bitcast<u32>(_e8081_) & 2147483647u));
                                                if (_e8248_ != _e8248_) {
                                                    phi_32194_ = true;
                                                } else {
                                                    phi_32194_ = (_e8260_ >= _e8248_);
                                                }
                                                let _e8273_ = phi_32194_;
                                                let _e8274_ = select(_e8248_, _e8260_, _e8273_);
                                                if (_e8251_ != _e8251_) {
                                                    phi_32209_ = true;
                                                } else {
                                                    phi_32209_ = (_e8263_ >= _e8251_);
                                                }
                                                let _e8278_ = phi_32209_;
                                                let _e8279_ = select(_e8251_, _e8263_, _e8278_);
                                                if (_e8254_ != _e8254_) {
                                                    phi_32224_ = true;
                                                } else {
                                                    phi_32224_ = (_e8266_ >= _e8254_);
                                                }
                                                let _e8283_ = phi_32224_;
                                                let _e8284_ = select(_e8254_, _e8266_, _e8283_);
                                                if (_e8257_ != _e8257_) {
                                                    phi_32239_ = true;
                                                } else {
                                                    phi_32239_ = (_e8269_ >= _e8257_);
                                                }
                                                let _e8288_ = phi_32239_;
                                                let _e8289_ = select(_e8257_, _e8269_, _e8288_);
                                                let _e8294_ = (1f - (_e8214_ + _e8218_));
                                                let _e8295_ = (1f - (_e8215_ + _e8219_));
                                                let _e8296_ = (1f - (_e8216_ + _e8220_));
                                                let _e8297_ = (1f - (_e8217_ + _e8221_));
                                                let _e8323_ = select(((_e8202_ * _e8294_) * ((_e8294_ * _e8294_) + ((4f * _e8214_) * _e8218_))), 0f, (_e8274_ <= 0f));
                                                let _e8325_ = select(((_e8203_ * _e8295_) * ((_e8295_ * _e8295_) + ((4f * _e8215_) * _e8219_))), 0f, (_e8279_ <= 0f));
                                                let _e8327_ = select(((_e8204_ * _e8296_) * ((_e8296_ * _e8296_) + ((4f * _e8216_) * _e8220_))), 0f, (_e8284_ <= 0f));
                                                let _e8329_ = select(((_e8205_ * _e8297_) * ((_e8297_ * _e8297_) + ((4f * _e8217_) * _e8221_))), 0f, (_e8289_ <= 0f));
                                                if (_e8274_ <= 1f) {
                                                    if (_e8279_ <= 1f) {
                                                        let _e8333_ = (_e8284_ <= 1f);
                                                        if _e8333_ {
                                                            phi_20327_ = (_e8289_ <= 1f);
                                                        } else {
                                                            phi_20327_ = bool();
                                                        }
                                                        let _e8336_ = phi_20327_;
                                                        phi_20329_ = _e8336_;
                                                        phi_20330_ = select(true, false, _e8333_);
                                                    } else {
                                                        phi_20329_ = bool();
                                                        phi_20330_ = true;
                                                    }
                                                    let _e8339_ = phi_20329_;
                                                    let _e8341_ = phi_20330_;
                                                    phi_20331_ = _e8339_;
                                                    phi_20332_ = _e8341_;
                                                } else {
                                                    phi_20331_ = bool();
                                                    phi_20332_ = true;
                                                }
                                                let _e8343_ = phi_20331_;
                                                let _e8345_ = phi_20332_;
                                                if select(_e8343_, false, _e8345_) {
                                                    phi_20569_ = type_32_(array<f32, 4>(_e8323_, _e8325_, _e8327_, _e8329_));
                                                } else {
                                                    if (_e8242_ != _e8242_) {
                                                        phi_32254_ = true;
                                                    } else {
                                                        phi_32254_ = (0.000001f >= _e8242_);
                                                    }
                                                    let _e8350_ = phi_32254_;
                                                    let _e8351_ = select(_e8242_, 0.000001f, _e8350_);
                                                    if (_e8243_ != _e8243_) {
                                                        phi_32269_ = true;
                                                    } else {
                                                        phi_32269_ = (0.000001f >= _e8243_);
                                                    }
                                                    let _e8355_ = phi_32269_;
                                                    let _e8356_ = select(_e8243_, 0.000001f, _e8355_);
                                                    if (_e8244_ != _e8244_) {
                                                        phi_32284_ = true;
                                                    } else {
                                                        phi_32284_ = (0.000001f >= _e8244_);
                                                    }
                                                    let _e8360_ = phi_32284_;
                                                    let _e8361_ = select(_e8244_, 0.000001f, _e8360_);
                                                    if (_e8245_ != _e8245_) {
                                                        phi_32299_ = true;
                                                    } else {
                                                        phi_32299_ = (0.000001f >= _e8245_);
                                                    }
                                                    let _e8365_ = phi_32299_;
                                                    let _e8366_ = select(_e8245_, 0.000001f, _e8365_);
                                                    let _e8369_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8351_)));
                                                    let _e8372_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8356_)));
                                                    let _e8375_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8361_)));
                                                    let _e8378_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8366_)));
                                                    let _e8388_ = (_e8369_ + (_e8369_ * (1f - (_e8351_ * _e8369_))));
                                                    let _e8390_ = (_e8372_ + (_e8372_ * (1f - (_e8356_ * _e8372_))));
                                                    let _e8392_ = (_e8375_ + (_e8375_ * (1f - (_e8361_ * _e8375_))));
                                                    let _e8394_ = (_e8378_ + (_e8378_ * (1f - (_e8366_ * _e8378_))));
                                                    phi_20569_ = type_32_(array<f32, 4>(select(select(select(_e8323_, ((select(_e8206_, 0f, _e8225_) * _e8323_) * (_e8388_ + (_e8388_ * (1f - (_e8351_ * _e8388_))))), (select((select(0f, 1f, (-1f <= _e8086_)) * select(0f, 1f, (_e8105_ <= 1f))), 1f, (_e8242_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8086_)), 0f, (_e8105_ <= -1f)), select(select(select(_e8325_, ((select(_e8207_, 0f, _e8230_) * _e8325_) * (_e8390_ + (_e8390_ * (1f - (_e8356_ * _e8390_))))), (select((select(0f, 1f, (-1f <= _e8091_)) * select(0f, 1f, (_e8109_ <= 1f))), 1f, (_e8243_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8091_)), 0f, (_e8109_ <= -1f)), select(select(select(_e8327_, ((select(_e8208_, 0f, _e8235_) * _e8327_) * (_e8392_ + (_e8392_ * (1f - (_e8361_ * _e8392_))))), (select((select(0f, 1f, (-1f <= _e8096_)) * select(0f, 1f, (_e8113_ <= 1f))), 1f, (_e8244_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8096_)), 0f, (_e8113_ <= -1f)), select(select(select(_e8329_, ((select(_e8209_, 0f, _e8240_) * _e8329_) * (_e8394_ + (_e8394_ * (1f - (_e8366_ * _e8394_))))), (select((select(0f, 1f, (-1f <= _e8101_)) * select(0f, 1f, (_e8117_ <= 1f))), 1f, (_e8245_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e8101_)), 0f, (_e8117_ <= -1f))));
                                                }
                                                let _e8475_ = phi_20569_;
                                                phi_20605_ = type_33_(type_32_(array<f32, 4>(((_e7869_ * (1f - _e7159_)) + (_e7159_ * _e8475_.member[0])), ((_e7871_ * (1f - _e7164_)) + (_e7164_ * _e8475_.member[1])), ((_e7873_ * (1f - _e7169_)) + (_e7169_ * _e8475_.member[2])), ((_e7875_ * (1f - _e7174_)) + (_e7174_ * _e8475_.member[3])))), type_32_(_e7921_), type_32_(_e7303_));
                                            }
                                            let _e8510_ = phi_20605_;
                                            let _e8513_ = local_20_;
                                            if _e7135_ {
                                                phi_32314_ = true;
                                            } else {
                                                phi_32314_ = (0f >= _e7128_);
                                            }
                                            let _e8520_ = phi_32314_;
                                            let _e8521_ = select(_e7128_, 0f, _e8520_);
                                            if _e7140_ {
                                                phi_32329_ = true;
                                            } else {
                                                phi_32329_ = (0f >= _e7130_);
                                            }
                                            let _e8524_ = phi_32329_;
                                            let _e8525_ = select(_e7130_, 0f, _e8524_);
                                            if _e7145_ {
                                                phi_32344_ = true;
                                            } else {
                                                phi_32344_ = (0f >= _e7132_);
                                            }
                                            let _e8528_ = phi_32344_;
                                            let _e8529_ = select(_e7132_, 0f, _e8528_);
                                            if _e7150_ {
                                                phi_32359_ = true;
                                            } else {
                                                phi_32359_ = (0f >= _e7134_);
                                            }
                                            let _e8532_ = phi_32359_;
                                            let _e8533_ = select(_e7134_, 0f, _e8532_);
                                            if (_e8521_ != _e8521_) {
                                                phi_32374_ = true;
                                            } else {
                                                phi_32374_ = (1f <= _e8521_);
                                            }
                                            let _e8537_ = phi_32374_;
                                            let _e8538_ = select(_e8521_, 1f, _e8537_);
                                            if (_e8525_ != _e8525_) {
                                                phi_32389_ = true;
                                            } else {
                                                phi_32389_ = (1f <= _e8525_);
                                            }
                                            let _e8542_ = phi_32389_;
                                            let _e8543_ = select(_e8525_, 1f, _e8542_);
                                            if (_e8529_ != _e8529_) {
                                                phi_32404_ = true;
                                            } else {
                                                phi_32404_ = (1f <= _e8529_);
                                            }
                                            let _e8547_ = phi_32404_;
                                            let _e8548_ = select(_e8529_, 1f, _e8547_);
                                            if (_e8533_ != _e8533_) {
                                                phi_32419_ = true;
                                            } else {
                                                phi_32419_ = (1f <= _e8533_);
                                            }
                                            let _e8552_ = phi_32419_;
                                            let _e8553_ = select(_e8533_, 1f, _e8552_);
                                            let _e8567_ = select(0f, _e8513_[0], (bitcast<f32>((bitcast<u32>(_e8513_[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8569_ = select(0f, _e8513_[1], (bitcast<f32>((bitcast<u32>(_e8513_[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8571_ = select(0f, _e8513_[2], (bitcast<f32>((bitcast<u32>(_e8513_[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8573_ = select(0f, _e8513_[3], (bitcast<f32>((bitcast<u32>(_e8513_[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8574_ = (_e8567_ != _e8567_);
                                            if _e8574_ {
                                                phi_32434_ = true;
                                            } else {
                                                phi_32434_ = (-4096f >= _e8567_);
                                            }
                                            let _e8577_ = phi_32434_;
                                            let _e8578_ = select(_e8567_, -4096f, _e8577_);
                                            let _e8579_ = (_e8569_ != _e8569_);
                                            if _e8579_ {
                                                phi_32449_ = true;
                                            } else {
                                                phi_32449_ = (-4096f >= _e8569_);
                                            }
                                            let _e8582_ = phi_32449_;
                                            let _e8583_ = select(_e8569_, -4096f, _e8582_);
                                            let _e8584_ = (_e8571_ != _e8571_);
                                            if _e8584_ {
                                                phi_32464_ = true;
                                            } else {
                                                phi_32464_ = (-4096f >= _e8571_);
                                            }
                                            let _e8587_ = phi_32464_;
                                            let _e8588_ = select(_e8571_, -4096f, _e8587_);
                                            let _e8589_ = (_e8573_ != _e8573_);
                                            if _e8589_ {
                                                phi_32479_ = true;
                                            } else {
                                                phi_32479_ = (-4096f >= _e8573_);
                                            }
                                            let _e8592_ = phi_32479_;
                                            let _e8593_ = select(_e8573_, -4096f, _e8592_);
                                            if (_e8578_ != _e8578_) {
                                                phi_32494_ = true;
                                            } else {
                                                phi_32494_ = (4096f <= _e8578_);
                                            }
                                            let _e8597_ = phi_32494_;
                                            let _e8598_ = select(_e8578_, 4096f, _e8597_);
                                            if (_e8583_ != _e8583_) {
                                                phi_32509_ = true;
                                            } else {
                                                phi_32509_ = (4096f <= _e8583_);
                                            }
                                            let _e8602_ = phi_32509_;
                                            let _e8603_ = select(_e8583_, 4096f, _e8602_);
                                            if (_e8588_ != _e8588_) {
                                                phi_32524_ = true;
                                            } else {
                                                phi_32524_ = (4096f <= _e8588_);
                                            }
                                            let _e8607_ = phi_32524_;
                                            let _e8608_ = select(_e8588_, 4096f, _e8607_);
                                            if (_e8593_ != _e8593_) {
                                                phi_32539_ = true;
                                            } else {
                                                phi_32539_ = (4096f <= _e8593_);
                                            }
                                            let _e8612_ = phi_32539_;
                                            let _e8613_ = select(_e8593_, 4096f, _e8612_);
                                            let _e8635_ = select(0f, _e7018_.member[0], (bitcast<f32>((bitcast<u32>(_e7018_.member[0]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8637_ = select(0f, _e7018_.member[1], (bitcast<f32>((bitcast<u32>(_e7018_.member[1]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8639_ = select(0f, _e7018_.member[2], (bitcast<f32>((bitcast<u32>(_e7018_.member[2]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e8641_ = select(0f, _e7018_.member[3], (bitcast<f32>((bitcast<u32>(_e7018_.member[3]) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e8635_ != _e8635_) {
                                                phi_32554_ = true;
                                            } else {
                                                phi_32554_ = (-4096f >= _e8635_);
                                            }
                                            let _e8645_ = phi_32554_;
                                            let _e8646_ = select(_e8635_, -4096f, _e8645_);
                                            if (_e8637_ != _e8637_) {
                                                phi_32569_ = true;
                                            } else {
                                                phi_32569_ = (-4096f >= _e8637_);
                                            }
                                            let _e8650_ = phi_32569_;
                                            let _e8651_ = select(_e8637_, -4096f, _e8650_);
                                            if (_e8639_ != _e8639_) {
                                                phi_32584_ = true;
                                            } else {
                                                phi_32584_ = (-4096f >= _e8639_);
                                            }
                                            let _e8655_ = phi_32584_;
                                            let _e8656_ = select(_e8639_, -4096f, _e8655_);
                                            if (_e8641_ != _e8641_) {
                                                phi_32599_ = true;
                                            } else {
                                                phi_32599_ = (-4096f >= _e8641_);
                                            }
                                            let _e8660_ = phi_32599_;
                                            let _e8661_ = select(_e8641_, -4096f, _e8660_);
                                            if (_e8646_ != _e8646_) {
                                                phi_32614_ = true;
                                            } else {
                                                phi_32614_ = (4096f <= _e8646_);
                                            }
                                            let _e8665_ = phi_32614_;
                                            let _e8666_ = select(_e8646_, 4096f, _e8665_);
                                            if (_e8651_ != _e8651_) {
                                                phi_32629_ = true;
                                            } else {
                                                phi_32629_ = (4096f <= _e8651_);
                                            }
                                            let _e8670_ = phi_32629_;
                                            let _e8671_ = select(_e8651_, 4096f, _e8670_);
                                            if (_e8656_ != _e8656_) {
                                                phi_32644_ = true;
                                            } else {
                                                phi_32644_ = (4096f <= _e8656_);
                                            }
                                            let _e8675_ = phi_32644_;
                                            let _e8676_ = select(_e8656_, 4096f, _e8675_);
                                            if (_e8661_ != _e8661_) {
                                                phi_32659_ = true;
                                            } else {
                                                phi_32659_ = (4096f <= _e8661_);
                                            }
                                            let _e8680_ = phi_32659_;
                                            let _e8681_ = select(_e8661_, 4096f, _e8680_);
                                            let _e8682_ = array<f32, 4>(_e8666_, _e8671_, _e8676_, _e8681_);
                                            let _e8683_ = (_e8666_ * _e8666_);
                                            let _e8684_ = (_e8671_ * _e8671_);
                                            let _e8685_ = (_e8676_ * _e8676_);
                                            let _e8686_ = (_e8681_ * _e8681_);
                                            let _e8687_ = (_e8598_ * _e8598_);
                                            let _e8688_ = (_e8603_ * _e8603_);
                                            let _e8689_ = (_e8608_ * _e8608_);
                                            let _e8690_ = (_e8613_ * _e8613_);
                                            let _e8691_ = (1f + _e8683_);
                                            let _e8692_ = (1f + _e8684_);
                                            let _e8693_ = (1f + _e8685_);
                                            let _e8694_ = (1f + _e8686_);
                                            let _e8695_ = (1f + _e8687_);
                                            let _e8696_ = (1f + _e8688_);
                                            let _e8697_ = (1f + _e8689_);
                                            let _e8698_ = (1f + _e8690_);
                                            if (_e8691_ != _e8691_) {
                                                phi_32674_ = true;
                                            } else {
                                                phi_32674_ = (0.000000000001f >= _e8691_);
                                            }
                                            let _e8702_ = phi_32674_;
                                            let _e8703_ = select(_e8691_, 0.000000000001f, _e8702_);
                                            if (_e8692_ != _e8692_) {
                                                phi_32689_ = true;
                                            } else {
                                                phi_32689_ = (0.000000000001f >= _e8692_);
                                            }
                                            let _e8707_ = phi_32689_;
                                            let _e8708_ = select(_e8692_, 0.000000000001f, _e8707_);
                                            if (_e8693_ != _e8693_) {
                                                phi_32704_ = true;
                                            } else {
                                                phi_32704_ = (0.000000000001f >= _e8693_);
                                            }
                                            let _e8712_ = phi_32704_;
                                            let _e8713_ = select(_e8693_, 0.000000000001f, _e8712_);
                                            if (_e8694_ != _e8694_) {
                                                phi_32719_ = true;
                                            } else {
                                                phi_32719_ = (0.000000000001f >= _e8694_);
                                            }
                                            let _e8717_ = phi_32719_;
                                            let _e8718_ = select(_e8694_, 0.000000000001f, _e8717_);
                                            let _e8719_ = (_e8703_ * 0.5f);
                                            let _e8720_ = (_e8708_ * 0.5f);
                                            let _e8721_ = (_e8713_ * 0.5f);
                                            let _e8722_ = (_e8718_ * 0.5f);
                                            let _e8727_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8703_) >> bitcast<u32>(1i))));
                                            let _e8732_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8708_) >> bitcast<u32>(1i))));
                                            let _e8737_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8713_) >> bitcast<u32>(1i))));
                                            let _e8742_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8718_) >> bitcast<u32>(1i))));
                                            let _e8755_ = (_e8727_ * (1.5f - (_e8719_ * (_e8727_ * _e8727_))));
                                            let _e8756_ = (_e8732_ * (1.5f - (_e8720_ * (_e8732_ * _e8732_))));
                                            let _e8757_ = (_e8737_ * (1.5f - (_e8721_ * (_e8737_ * _e8737_))));
                                            let _e8758_ = (_e8742_ * (1.5f - (_e8722_ * (_e8742_ * _e8742_))));
                                            let _e8771_ = (_e8755_ * (1.5f - (_e8719_ * (_e8755_ * _e8755_))));
                                            let _e8772_ = (_e8756_ * (1.5f - (_e8720_ * (_e8756_ * _e8756_))));
                                            let _e8773_ = (_e8757_ * (1.5f - (_e8721_ * (_e8757_ * _e8757_))));
                                            let _e8774_ = (_e8758_ * (1.5f - (_e8722_ * (_e8758_ * _e8758_))));
                                            let _e8787_ = (_e8771_ * (1.5f - (_e8719_ * (_e8771_ * _e8771_))));
                                            let _e8788_ = (_e8772_ * (1.5f - (_e8720_ * (_e8772_ * _e8772_))));
                                            let _e8789_ = (_e8773_ * (1.5f - (_e8721_ * (_e8773_ * _e8773_))));
                                            let _e8790_ = (_e8774_ * (1.5f - (_e8722_ * (_e8774_ * _e8774_))));
                                            if (_e8695_ != _e8695_) {
                                                phi_32734_ = true;
                                            } else {
                                                phi_32734_ = (0.000000000001f >= _e8695_);
                                            }
                                            let _e8794_ = phi_32734_;
                                            let _e8795_ = select(_e8695_, 0.000000000001f, _e8794_);
                                            if (_e8696_ != _e8696_) {
                                                phi_32749_ = true;
                                            } else {
                                                phi_32749_ = (0.000000000001f >= _e8696_);
                                            }
                                            let _e8799_ = phi_32749_;
                                            let _e8800_ = select(_e8696_, 0.000000000001f, _e8799_);
                                            if (_e8697_ != _e8697_) {
                                                phi_32764_ = true;
                                            } else {
                                                phi_32764_ = (0.000000000001f >= _e8697_);
                                            }
                                            let _e8804_ = phi_32764_;
                                            let _e8805_ = select(_e8697_, 0.000000000001f, _e8804_);
                                            if (_e8698_ != _e8698_) {
                                                phi_32779_ = true;
                                            } else {
                                                phi_32779_ = (0.000000000001f >= _e8698_);
                                            }
                                            let _e8809_ = phi_32779_;
                                            let _e8810_ = select(_e8698_, 0.000000000001f, _e8809_);
                                            let _e8811_ = (_e8795_ * 0.5f);
                                            let _e8812_ = (_e8800_ * 0.5f);
                                            let _e8813_ = (_e8805_ * 0.5f);
                                            let _e8814_ = (_e8810_ * 0.5f);
                                            let _e8819_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8795_) >> bitcast<u32>(1i))));
                                            let _e8824_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8800_) >> bitcast<u32>(1i))));
                                            let _e8829_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8805_) >> bitcast<u32>(1i))));
                                            let _e8834_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e8810_) >> bitcast<u32>(1i))));
                                            let _e8847_ = (_e8819_ * (1.5f - (_e8811_ * (_e8819_ * _e8819_))));
                                            let _e8848_ = (_e8824_ * (1.5f - (_e8812_ * (_e8824_ * _e8824_))));
                                            let _e8849_ = (_e8829_ * (1.5f - (_e8813_ * (_e8829_ * _e8829_))));
                                            let _e8850_ = (_e8834_ * (1.5f - (_e8814_ * (_e8834_ * _e8834_))));
                                            let _e8863_ = (_e8847_ * (1.5f - (_e8811_ * (_e8847_ * _e8847_))));
                                            let _e8864_ = (_e8848_ * (1.5f - (_e8812_ * (_e8848_ * _e8848_))));
                                            let _e8865_ = (_e8849_ * (1.5f - (_e8813_ * (_e8849_ * _e8849_))));
                                            let _e8866_ = (_e8850_ * (1.5f - (_e8814_ * (_e8850_ * _e8850_))));
                                            let _e8879_ = (_e8863_ * (1.5f - (_e8811_ * (_e8863_ * _e8863_))));
                                            let _e8880_ = (_e8864_ * (1.5f - (_e8812_ * (_e8864_ * _e8864_))));
                                            let _e8881_ = (_e8865_ * (1.5f - (_e8813_ * (_e8865_ * _e8865_))));
                                            let _e8882_ = (_e8866_ * (1.5f - (_e8814_ * (_e8866_ * _e8866_))));
                                            let _e8883_ = (_e8691_ * _e8787_);
                                            let _e8884_ = (_e8692_ * _e8788_);
                                            let _e8885_ = (_e8693_ * _e8789_);
                                            let _e8886_ = (_e8694_ * _e8790_);
                                            let _e8887_ = (_e8695_ * _e8879_);
                                            let _e8888_ = (_e8696_ * _e8880_);
                                            let _e8889_ = (_e8697_ * _e8881_);
                                            let _e8890_ = (_e8698_ * _e8882_);
                                            let _e8891_ = (_e8883_ + 1f);
                                            let _e8892_ = (_e8884_ + 1f);
                                            let _e8893_ = (_e8885_ + 1f);
                                            let _e8894_ = (_e8886_ + 1f);
                                            let _e8897_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8891_)));
                                            let _e8900_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8892_)));
                                            let _e8903_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8893_)));
                                            let _e8906_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8894_)));
                                            let _e8916_ = (_e8897_ + (_e8897_ * (1f - (_e8891_ * _e8897_))));
                                            let _e8918_ = (_e8900_ + (_e8900_ * (1f - (_e8892_ * _e8900_))));
                                            let _e8920_ = (_e8903_ + (_e8903_ * (1f - (_e8893_ * _e8903_))));
                                            let _e8922_ = (_e8906_ + (_e8906_ * (1f - (_e8894_ * _e8906_))));
                                            let _e8932_ = (_e8916_ + (_e8916_ * (1f - (_e8891_ * _e8916_))));
                                            let _e8934_ = (_e8918_ + (_e8918_ * (1f - (_e8892_ * _e8918_))));
                                            let _e8936_ = (_e8920_ + (_e8920_ * (1f - (_e8893_ * _e8920_))));
                                            let _e8938_ = (_e8922_ + (_e8922_ * (1f - (_e8894_ * _e8922_))));
                                            let _e8955_ = (_e8683_ * (_e8932_ + (_e8932_ * (1f - (_e8891_ * _e8932_)))));
                                            let _e8956_ = (_e8684_ * (_e8934_ + (_e8934_ * (1f - (_e8892_ * _e8934_)))));
                                            let _e8957_ = (_e8685_ * (_e8936_ + (_e8936_ * (1f - (_e8893_ * _e8936_)))));
                                            let _e8958_ = (_e8686_ * (_e8938_ + (_e8938_ * (1f - (_e8894_ * _e8938_)))));
                                            let _e8959_ = (_e8887_ + 1f);
                                            let _e8960_ = (_e8888_ + 1f);
                                            let _e8961_ = (_e8889_ + 1f);
                                            let _e8962_ = (_e8890_ + 1f);
                                            let _e8965_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8959_)));
                                            let _e8968_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8960_)));
                                            let _e8971_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8961_)));
                                            let _e8974_ = bitcast<f32>((2129859011u - bitcast<u32>(_e8962_)));
                                            let _e8984_ = (_e8965_ + (_e8965_ * (1f - (_e8959_ * _e8965_))));
                                            let _e8986_ = (_e8968_ + (_e8968_ * (1f - (_e8960_ * _e8968_))));
                                            let _e8988_ = (_e8971_ + (_e8971_ * (1f - (_e8961_ * _e8971_))));
                                            let _e8990_ = (_e8974_ + (_e8974_ * (1f - (_e8962_ * _e8974_))));
                                            let _e9000_ = (_e8984_ + (_e8984_ * (1f - (_e8959_ * _e8984_))));
                                            let _e9002_ = (_e8986_ + (_e8986_ * (1f - (_e8960_ * _e8986_))));
                                            let _e9004_ = (_e8988_ + (_e8988_ * (1f - (_e8961_ * _e8988_))));
                                            let _e9006_ = (_e8990_ + (_e8990_ * (1f - (_e8962_ * _e8990_))));
                                            let _e9023_ = (_e8687_ * (_e9000_ + (_e9000_ * (1f - (_e8959_ * _e9000_)))));
                                            let _e9024_ = (_e8688_ * (_e9002_ + (_e9002_ * (1f - (_e8960_ * _e9002_)))));
                                            let _e9025_ = (_e8689_ * (_e9004_ + (_e9004_ * (1f - (_e8961_ * _e9004_)))));
                                            let _e9026_ = (_e8690_ * (_e9006_ + (_e9006_ * (1f - (_e8962_ * _e9006_)))));
                                            let _e9027_ = (_e8883_ + _e8887_);
                                            let _e9028_ = (_e8884_ + _e8888_);
                                            let _e9029_ = (_e8885_ + _e8889_);
                                            let _e9030_ = (_e8886_ + _e8890_);
                                            let _e9033_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9027_)));
                                            let _e9036_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9028_)));
                                            let _e9039_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9029_)));
                                            let _e9042_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9030_)));
                                            let _e9052_ = (_e9033_ + (_e9033_ * (1f - (_e9027_ * _e9033_))));
                                            let _e9054_ = (_e9036_ + (_e9036_ * (1f - (_e9028_ * _e9036_))));
                                            let _e9056_ = (_e9039_ + (_e9039_ * (1f - (_e9029_ * _e9039_))));
                                            let _e9058_ = (_e9042_ + (_e9042_ * (1f - (_e9030_ * _e9042_))));
                                            let _e9068_ = (_e9052_ + (_e9052_ * (1f - (_e9027_ * _e9052_))));
                                            let _e9070_ = (_e9054_ + (_e9054_ * (1f - (_e9028_ * _e9054_))));
                                            let _e9072_ = (_e9056_ + (_e9056_ * (1f - (_e9029_ * _e9056_))));
                                            let _e9074_ = (_e9058_ + (_e9058_ * (1f - (_e9030_ * _e9058_))));
                                            let _e9084_ = (_e9068_ + (_e9068_ * (1f - (_e9027_ * _e9068_))));
                                            let _e9086_ = (_e9070_ + (_e9070_ * (1f - (_e9028_ * _e9070_))));
                                            let _e9088_ = (_e9072_ + (_e9072_ * (1f - (_e9029_ * _e9072_))));
                                            let _e9090_ = (_e9074_ + (_e9074_ * (1f - (_e9030_ * _e9074_))));
                                            let _e9111_ = (((_e8666_ + _e8598_) * _e9084_) * (1f + (0.5f * (_e8787_ * _e8879_))));
                                            let _e9112_ = (((_e8671_ + _e8603_) * _e9086_) * (1f + (0.5f * (_e8788_ * _e8880_))));
                                            let _e9113_ = (((_e8676_ + _e8608_) * _e9088_) * (1f + (0.5f * (_e8789_ * _e8881_))));
                                            let _e9114_ = (((_e8681_ + _e8613_) * _e9090_) * (1f + (0.5f * (_e8790_ * _e8882_))));
                                            let _e9115_ = (_e8955_ + _e9023_);
                                            let _e9116_ = (_e8956_ + _e9024_);
                                            let _e9117_ = (_e8957_ + _e9025_);
                                            let _e9118_ = (_e8958_ + _e9026_);
                                            let _e9159_ = ((((((_e9115_ * (2f + _e9115_)) + ((_e8955_ * _e9023_) * (1f + _e9115_))) + (_e8666_ * _e8598_)) * _e9084_) * _e8787_) * _e8879_);
                                            let _e9160_ = ((((((_e9116_ * (2f + _e9116_)) + ((_e8956_ * _e9024_) * (1f + _e9116_))) + (_e8671_ * _e8603_)) * _e9086_) * _e8788_) * _e8880_);
                                            let _e9161_ = ((((((_e9117_ * (2f + _e9117_)) + ((_e8957_ * _e9025_) * (1f + _e9117_))) + (_e8676_ * _e8608_)) * _e9088_) * _e8789_) * _e8881_);
                                            let _e9162_ = ((((((_e9118_ * (2f + _e9118_)) + ((_e8958_ * _e9026_) * (1f + _e9118_))) + (_e8681_ * _e8613_)) * _e9090_) * _e8790_) * _e8882_);
                                            let _e9165_ = bitcast<f32>((bitcast<u32>(_e8598_) & 2147483647u));
                                            let _e9168_ = bitcast<f32>((bitcast<u32>(_e8603_) & 2147483647u));
                                            let _e9171_ = bitcast<f32>((bitcast<u32>(_e8608_) & 2147483647u));
                                            let _e9174_ = bitcast<f32>((bitcast<u32>(_e8613_) & 2147483647u));
                                            let _e9177_ = bitcast<f32>((bitcast<u32>(_e8666_) & 2147483647u));
                                            let _e9180_ = bitcast<f32>((bitcast<u32>(_e8671_) & 2147483647u));
                                            let _e9183_ = bitcast<f32>((bitcast<u32>(_e8676_) & 2147483647u));
                                            let _e9186_ = bitcast<f32>((bitcast<u32>(_e8681_) & 2147483647u));
                                            if (_e9165_ != _e9165_) {
                                                phi_32794_ = true;
                                            } else {
                                                phi_32794_ = (_e9177_ >= _e9165_);
                                            }
                                            let _e9190_ = phi_32794_;
                                            if (_e9168_ != _e9168_) {
                                                phi_32809_ = true;
                                            } else {
                                                phi_32809_ = (_e9180_ >= _e9168_);
                                            }
                                            let _e9195_ = phi_32809_;
                                            if (_e9171_ != _e9171_) {
                                                phi_32824_ = true;
                                            } else {
                                                phi_32824_ = (_e9183_ >= _e9171_);
                                            }
                                            let _e9200_ = phi_32824_;
                                            if (_e9174_ != _e9174_) {
                                                phi_32839_ = true;
                                            } else {
                                                phi_32839_ = (_e9186_ >= _e9174_);
                                            }
                                            let _e9205_ = phi_32839_;
                                            if (_e9111_ != _e9111_) {
                                                phi_32854_ = true;
                                            } else {
                                                phi_32854_ = (-1f >= _e9111_);
                                            }
                                            let _e9210_ = phi_32854_;
                                            let _e9211_ = select(_e9111_, -1f, _e9210_);
                                            if (_e9112_ != _e9112_) {
                                                phi_32869_ = true;
                                            } else {
                                                phi_32869_ = (-1f >= _e9112_);
                                            }
                                            let _e9215_ = phi_32869_;
                                            let _e9216_ = select(_e9112_, -1f, _e9215_);
                                            if (_e9113_ != _e9113_) {
                                                phi_32884_ = true;
                                            } else {
                                                phi_32884_ = (-1f >= _e9113_);
                                            }
                                            let _e9220_ = phi_32884_;
                                            let _e9221_ = select(_e9113_, -1f, _e9220_);
                                            if (_e9114_ != _e9114_) {
                                                phi_32899_ = true;
                                            } else {
                                                phi_32899_ = (-1f >= _e9114_);
                                            }
                                            let _e9225_ = phi_32899_;
                                            let _e9226_ = select(_e9114_, -1f, _e9225_);
                                            if (_e9211_ != _e9211_) {
                                                phi_32914_ = true;
                                            } else {
                                                phi_32914_ = (1f <= _e9211_);
                                            }
                                            let _e9230_ = phi_32914_;
                                            if (_e9216_ != _e9216_) {
                                                phi_32929_ = true;
                                            } else {
                                                phi_32929_ = (1f <= _e9216_);
                                            }
                                            let _e9235_ = phi_32929_;
                                            if (_e9221_ != _e9221_) {
                                                phi_32944_ = true;
                                            } else {
                                                phi_32944_ = (1f <= _e9221_);
                                            }
                                            let _e9240_ = phi_32944_;
                                            if (_e9226_ != _e9226_) {
                                                phi_32959_ = true;
                                            } else {
                                                phi_32959_ = (1f <= _e9226_);
                                            }
                                            let _e9245_ = phi_32959_;
                                            let _e9247_ = (select(_e9165_, _e9177_, _e9190_) <= 0f);
                                            let _e9248_ = select(select(_e9211_, 1f, _e9230_), 0f, _e9247_);
                                            let _e9249_ = (select(_e9168_, _e9180_, _e9195_) <= 0f);
                                            let _e9250_ = select(select(_e9216_, 1f, _e9235_), 0f, _e9249_);
                                            let _e9251_ = (select(_e9171_, _e9183_, _e9200_) <= 0f);
                                            let _e9252_ = select(select(_e9221_, 1f, _e9240_), 0f, _e9251_);
                                            let _e9253_ = (select(_e9174_, _e9186_, _e9205_) <= 0f);
                                            let _e9254_ = select(select(_e9226_, 1f, _e9245_), 0f, _e9253_);
                                            if (_e9159_ != _e9159_) {
                                                phi_32974_ = true;
                                            } else {
                                                phi_32974_ = (0f >= _e9159_);
                                            }
                                            let _e9259_ = phi_32974_;
                                            let _e9260_ = select(_e9159_, 0f, _e9259_);
                                            if (_e9160_ != _e9160_) {
                                                phi_32989_ = true;
                                            } else {
                                                phi_32989_ = (0f >= _e9160_);
                                            }
                                            let _e9264_ = phi_32989_;
                                            let _e9265_ = select(_e9160_, 0f, _e9264_);
                                            if (_e9161_ != _e9161_) {
                                                phi_33004_ = true;
                                            } else {
                                                phi_33004_ = (0f >= _e9161_);
                                            }
                                            let _e9269_ = phi_33004_;
                                            let _e9270_ = select(_e9161_, 0f, _e9269_);
                                            if (_e9162_ != _e9162_) {
                                                phi_33019_ = true;
                                            } else {
                                                phi_33019_ = (0f >= _e9162_);
                                            }
                                            let _e9274_ = phi_33019_;
                                            let _e9275_ = select(_e9162_, 0f, _e9274_);
                                            if (_e9260_ != _e9260_) {
                                                phi_33034_ = true;
                                            } else {
                                                phi_33034_ = (1f <= _e9260_);
                                            }
                                            let _e9279_ = phi_33034_;
                                            if (_e9265_ != _e9265_) {
                                                phi_33049_ = true;
                                            } else {
                                                phi_33049_ = (1f <= _e9265_);
                                            }
                                            let _e9284_ = phi_33049_;
                                            if (_e9270_ != _e9270_) {
                                                phi_33064_ = true;
                                            } else {
                                                phi_33064_ = (1f <= _e9270_);
                                            }
                                            let _e9289_ = phi_33064_;
                                            if (_e9275_ != _e9275_) {
                                                phi_33079_ = true;
                                            } else {
                                                phi_33079_ = (1f <= _e9275_);
                                            }
                                            let _e9294_ = phi_33079_;
                                            let _e9300_ = array<f32, 4>(select(select(_e9260_, 1f, _e9279_), 0f, _e9247_), select(select(_e9265_, 1f, _e9284_), 0f, _e9249_), select(select(_e9270_, 1f, _e9289_), 0f, _e9251_), select(select(_e9275_, 1f, _e9294_), 0f, _e9253_));
                                            if (_e8538_ <= 0f) {
                                                if (_e8543_ <= 0f) {
                                                    let _e9303_ = (_e8548_ <= 0f);
                                                    if _e9303_ {
                                                        phi_21324_ = (_e8553_ <= 0f);
                                                    } else {
                                                        phi_21324_ = bool();
                                                    }
                                                    let _e9306_ = phi_21324_;
                                                    phi_21326_ = _e9306_;
                                                    phi_21327_ = select(true, false, _e9303_);
                                                } else {
                                                    phi_21326_ = bool();
                                                    phi_21327_ = true;
                                                }
                                                let _e9309_ = phi_21326_;
                                                let _e9311_ = phi_21327_;
                                                phi_21328_ = _e9309_;
                                                phi_21329_ = _e9311_;
                                            } else {
                                                phi_21328_ = bool();
                                                phi_21329_ = true;
                                            }
                                            let _e9313_ = phi_21328_;
                                            let _e9315_ = phi_21329_;
                                            if select(_e9313_, false, _e9315_) {
                                                phi_22017_ = type_33_(type_32_(array<f32, 4>(_e9248_, _e9250_, _e9252_, _e9254_)), type_32_(_e9300_), type_32_(_e8682_));
                                            } else {
                                                if _e8574_ {
                                                    phi_33094_ = true;
                                                } else {
                                                    phi_33094_ = (-4096f >= _e8567_);
                                                }
                                                let _e9319_ = phi_33094_;
                                                let _e9320_ = select(_e8567_, -4096f, _e9319_);
                                                if _e8579_ {
                                                    phi_33109_ = true;
                                                } else {
                                                    phi_33109_ = (-4096f >= _e8569_);
                                                }
                                                let _e9323_ = phi_33109_;
                                                let _e9324_ = select(_e8569_, -4096f, _e9323_);
                                                if _e8584_ {
                                                    phi_33124_ = true;
                                                } else {
                                                    phi_33124_ = (-4096f >= _e8571_);
                                                }
                                                let _e9327_ = phi_33124_;
                                                let _e9328_ = select(_e8571_, -4096f, _e9327_);
                                                if _e8589_ {
                                                    phi_33139_ = true;
                                                } else {
                                                    phi_33139_ = (-4096f >= _e8573_);
                                                }
                                                let _e9331_ = phi_33139_;
                                                let _e9332_ = select(_e8573_, -4096f, _e9331_);
                                                if (_e9320_ != _e9320_) {
                                                    phi_33154_ = true;
                                                } else {
                                                    phi_33154_ = (4096f <= _e9320_);
                                                }
                                                let _e9336_ = phi_33154_;
                                                let _e9337_ = select(_e9320_, 4096f, _e9336_);
                                                if (_e9324_ != _e9324_) {
                                                    phi_33169_ = true;
                                                } else {
                                                    phi_33169_ = (4096f <= _e9324_);
                                                }
                                                let _e9341_ = phi_33169_;
                                                let _e9342_ = select(_e9324_, 4096f, _e9341_);
                                                if (_e9328_ != _e9328_) {
                                                    phi_33184_ = true;
                                                } else {
                                                    phi_33184_ = (4096f <= _e9328_);
                                                }
                                                let _e9346_ = phi_33184_;
                                                let _e9347_ = select(_e9328_, 4096f, _e9346_);
                                                if (_e9332_ != _e9332_) {
                                                    phi_33199_ = true;
                                                } else {
                                                    phi_33199_ = (4096f <= _e9332_);
                                                }
                                                let _e9351_ = phi_33199_;
                                                let _e9352_ = select(_e9332_, 4096f, _e9351_);
                                                let _e9366_ = select(0f, _e9337_, (bitcast<f32>((bitcast<u32>(_e9337_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                let _e9368_ = select(0f, _e9342_, (bitcast<f32>((bitcast<u32>(_e9342_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                let _e9370_ = select(0f, _e9347_, (bitcast<f32>((bitcast<u32>(_e9347_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                let _e9372_ = select(0f, _e9352_, (bitcast<f32>((bitcast<u32>(_e9352_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                                if (_e9366_ != _e9366_) {
                                                    phi_33214_ = true;
                                                } else {
                                                    phi_33214_ = (-1000000000000000000f >= _e9366_);
                                                }
                                                let _e9376_ = phi_33214_;
                                                let _e9377_ = select(_e9366_, -1000000000000000000f, _e9376_);
                                                if (_e9368_ != _e9368_) {
                                                    phi_33229_ = true;
                                                } else {
                                                    phi_33229_ = (-1000000000000000000f >= _e9368_);
                                                }
                                                let _e9381_ = phi_33229_;
                                                let _e9382_ = select(_e9368_, -1000000000000000000f, _e9381_);
                                                if (_e9370_ != _e9370_) {
                                                    phi_33244_ = true;
                                                } else {
                                                    phi_33244_ = (-1000000000000000000f >= _e9370_);
                                                }
                                                let _e9386_ = phi_33244_;
                                                let _e9387_ = select(_e9370_, -1000000000000000000f, _e9386_);
                                                if (_e9372_ != _e9372_) {
                                                    phi_33259_ = true;
                                                } else {
                                                    phi_33259_ = (-1000000000000000000f >= _e9372_);
                                                }
                                                let _e9391_ = phi_33259_;
                                                let _e9392_ = select(_e9372_, -1000000000000000000f, _e9391_);
                                                if (_e9377_ != _e9377_) {
                                                    phi_33274_ = true;
                                                } else {
                                                    phi_33274_ = (1000000000000000000f <= _e9377_);
                                                }
                                                let _e9396_ = phi_33274_;
                                                let _e9397_ = select(_e9377_, 1000000000000000000f, _e9396_);
                                                if (_e9382_ != _e9382_) {
                                                    phi_33289_ = true;
                                                } else {
                                                    phi_33289_ = (1000000000000000000f <= _e9382_);
                                                }
                                                let _e9401_ = phi_33289_;
                                                let _e9402_ = select(_e9382_, 1000000000000000000f, _e9401_);
                                                if (_e9387_ != _e9387_) {
                                                    phi_33304_ = true;
                                                } else {
                                                    phi_33304_ = (1000000000000000000f <= _e9387_);
                                                }
                                                let _e9406_ = phi_33304_;
                                                let _e9407_ = select(_e9387_, 1000000000000000000f, _e9406_);
                                                if (_e9392_ != _e9392_) {
                                                    phi_33319_ = true;
                                                } else {
                                                    phi_33319_ = (1000000000000000000f <= _e9392_);
                                                }
                                                let _e9411_ = phi_33319_;
                                                let _e9412_ = select(_e9392_, 1000000000000000000f, _e9411_);
                                                let _e9414_ = select(0f, _e8666_, (_e9177_ <= 340282350000000000000000000000000000000f));
                                                let _e9416_ = select(0f, _e8671_, (_e9180_ <= 340282350000000000000000000000000000000f));
                                                let _e9418_ = select(0f, _e8676_, (_e9183_ <= 340282350000000000000000000000000000000f));
                                                let _e9420_ = select(0f, _e8681_, (_e9186_ <= 340282350000000000000000000000000000000f));
                                                if (_e9414_ != _e9414_) {
                                                    phi_33334_ = true;
                                                } else {
                                                    phi_33334_ = (-1000000000000000000f >= _e9414_);
                                                }
                                                let _e9424_ = phi_33334_;
                                                let _e9425_ = select(_e9414_, -1000000000000000000f, _e9424_);
                                                if (_e9416_ != _e9416_) {
                                                    phi_33349_ = true;
                                                } else {
                                                    phi_33349_ = (-1000000000000000000f >= _e9416_);
                                                }
                                                let _e9429_ = phi_33349_;
                                                let _e9430_ = select(_e9416_, -1000000000000000000f, _e9429_);
                                                if (_e9418_ != _e9418_) {
                                                    phi_33364_ = true;
                                                } else {
                                                    phi_33364_ = (-1000000000000000000f >= _e9418_);
                                                }
                                                let _e9434_ = phi_33364_;
                                                let _e9435_ = select(_e9418_, -1000000000000000000f, _e9434_);
                                                if (_e9420_ != _e9420_) {
                                                    phi_33379_ = true;
                                                } else {
                                                    phi_33379_ = (-1000000000000000000f >= _e9420_);
                                                }
                                                let _e9439_ = phi_33379_;
                                                let _e9440_ = select(_e9420_, -1000000000000000000f, _e9439_);
                                                if (_e9425_ != _e9425_) {
                                                    phi_33394_ = true;
                                                } else {
                                                    phi_33394_ = (1000000000000000000f <= _e9425_);
                                                }
                                                let _e9444_ = phi_33394_;
                                                let _e9445_ = select(_e9425_, 1000000000000000000f, _e9444_);
                                                if (_e9430_ != _e9430_) {
                                                    phi_33409_ = true;
                                                } else {
                                                    phi_33409_ = (1000000000000000000f <= _e9430_);
                                                }
                                                let _e9449_ = phi_33409_;
                                                let _e9450_ = select(_e9430_, 1000000000000000000f, _e9449_);
                                                if (_e9435_ != _e9435_) {
                                                    phi_33424_ = true;
                                                } else {
                                                    phi_33424_ = (1000000000000000000f <= _e9435_);
                                                }
                                                let _e9454_ = phi_33424_;
                                                let _e9455_ = select(_e9435_, 1000000000000000000f, _e9454_);
                                                if (_e9440_ != _e9440_) {
                                                    phi_33439_ = true;
                                                } else {
                                                    phi_33439_ = (1000000000000000000f <= _e9440_);
                                                }
                                                let _e9459_ = phi_33439_;
                                                let _e9460_ = select(_e9440_, 1000000000000000000f, _e9459_);
                                                let _e9461_ = (_e9397_ != _e9397_);
                                                if _e9461_ {
                                                    phi_33454_ = true;
                                                } else {
                                                    phi_33454_ = (_e9445_ <= _e9397_);
                                                }
                                                let _e9464_ = phi_33454_;
                                                let _e9465_ = select(_e9397_, _e9445_, _e9464_);
                                                let _e9466_ = (_e9402_ != _e9402_);
                                                if _e9466_ {
                                                    phi_33469_ = true;
                                                } else {
                                                    phi_33469_ = (_e9450_ <= _e9402_);
                                                }
                                                let _e9469_ = phi_33469_;
                                                let _e9470_ = select(_e9402_, _e9450_, _e9469_);
                                                let _e9471_ = (_e9407_ != _e9407_);
                                                if _e9471_ {
                                                    phi_33484_ = true;
                                                } else {
                                                    phi_33484_ = (_e9455_ <= _e9407_);
                                                }
                                                let _e9474_ = phi_33484_;
                                                let _e9475_ = select(_e9407_, _e9455_, _e9474_);
                                                let _e9476_ = (_e9412_ != _e9412_);
                                                if _e9476_ {
                                                    phi_33499_ = true;
                                                } else {
                                                    phi_33499_ = (_e9460_ <= _e9412_);
                                                }
                                                let _e9479_ = phi_33499_;
                                                let _e9480_ = select(_e9412_, _e9460_, _e9479_);
                                                if _e9461_ {
                                                    phi_33514_ = true;
                                                } else {
                                                    phi_33514_ = (_e9445_ >= _e9397_);
                                                }
                                                let _e9483_ = phi_33514_;
                                                let _e9484_ = select(_e9397_, _e9445_, _e9483_);
                                                if _e9466_ {
                                                    phi_33529_ = true;
                                                } else {
                                                    phi_33529_ = (_e9450_ >= _e9402_);
                                                }
                                                let _e9487_ = phi_33529_;
                                                let _e9488_ = select(_e9402_, _e9450_, _e9487_);
                                                if _e9471_ {
                                                    phi_33544_ = true;
                                                } else {
                                                    phi_33544_ = (_e9455_ >= _e9407_);
                                                }
                                                let _e9491_ = phi_33544_;
                                                let _e9492_ = select(_e9407_, _e9455_, _e9491_);
                                                if _e9476_ {
                                                    phi_33559_ = true;
                                                } else {
                                                    phi_33559_ = (_e9460_ >= _e9412_);
                                                }
                                                let _e9495_ = phi_33559_;
                                                let _e9496_ = select(_e9412_, _e9460_, _e9495_);
                                                if (_e9465_ != _e9465_) {
                                                    phi_33574_ = true;
                                                } else {
                                                    phi_33574_ = (-1f >= _e9465_);
                                                }
                                                let _e9500_ = phi_33574_;
                                                let _e9501_ = select(_e9465_, -1f, _e9500_);
                                                if (_e9470_ != _e9470_) {
                                                    phi_33589_ = true;
                                                } else {
                                                    phi_33589_ = (-1f >= _e9470_);
                                                }
                                                let _e9505_ = phi_33589_;
                                                let _e9506_ = select(_e9470_, -1f, _e9505_);
                                                if (_e9475_ != _e9475_) {
                                                    phi_33604_ = true;
                                                } else {
                                                    phi_33604_ = (-1f >= _e9475_);
                                                }
                                                let _e9510_ = phi_33604_;
                                                let _e9511_ = select(_e9475_, -1f, _e9510_);
                                                if (_e9480_ != _e9480_) {
                                                    phi_33619_ = true;
                                                } else {
                                                    phi_33619_ = (-1f >= _e9480_);
                                                }
                                                let _e9515_ = phi_33619_;
                                                let _e9516_ = select(_e9480_, -1f, _e9515_);
                                                if (_e9501_ != _e9501_) {
                                                    phi_33634_ = true;
                                                } else {
                                                    phi_33634_ = (1f <= _e9501_);
                                                }
                                                let _e9520_ = phi_33634_;
                                                let _e9521_ = select(_e9501_, 1f, _e9520_);
                                                if (_e9506_ != _e9506_) {
                                                    phi_33649_ = true;
                                                } else {
                                                    phi_33649_ = (1f <= _e9506_);
                                                }
                                                let _e9525_ = phi_33649_;
                                                let _e9526_ = select(_e9506_, 1f, _e9525_);
                                                if (_e9511_ != _e9511_) {
                                                    phi_33664_ = true;
                                                } else {
                                                    phi_33664_ = (1f <= _e9511_);
                                                }
                                                let _e9530_ = phi_33664_;
                                                let _e9531_ = select(_e9511_, 1f, _e9530_);
                                                if (_e9516_ != _e9516_) {
                                                    phi_33679_ = true;
                                                } else {
                                                    phi_33679_ = (1f <= _e9516_);
                                                }
                                                let _e9535_ = phi_33679_;
                                                let _e9536_ = select(_e9516_, 1f, _e9535_);
                                                if (_e9484_ != _e9484_) {
                                                    phi_33694_ = true;
                                                } else {
                                                    phi_33694_ = (-1f >= _e9484_);
                                                }
                                                let _e9540_ = phi_33694_;
                                                let _e9541_ = select(_e9484_, -1f, _e9540_);
                                                if (_e9488_ != _e9488_) {
                                                    phi_33709_ = true;
                                                } else {
                                                    phi_33709_ = (-1f >= _e9488_);
                                                }
                                                let _e9545_ = phi_33709_;
                                                let _e9546_ = select(_e9488_, -1f, _e9545_);
                                                if (_e9492_ != _e9492_) {
                                                    phi_33724_ = true;
                                                } else {
                                                    phi_33724_ = (-1f >= _e9492_);
                                                }
                                                let _e9550_ = phi_33724_;
                                                let _e9551_ = select(_e9492_, -1f, _e9550_);
                                                if (_e9496_ != _e9496_) {
                                                    phi_33739_ = true;
                                                } else {
                                                    phi_33739_ = (-1f >= _e9496_);
                                                }
                                                let _e9555_ = phi_33739_;
                                                let _e9556_ = select(_e9496_, -1f, _e9555_);
                                                if (_e9541_ != _e9541_) {
                                                    phi_33754_ = true;
                                                } else {
                                                    phi_33754_ = (1f <= _e9541_);
                                                }
                                                let _e9560_ = phi_33754_;
                                                let _e9561_ = select(_e9541_, 1f, _e9560_);
                                                if (_e9546_ != _e9546_) {
                                                    phi_33769_ = true;
                                                } else {
                                                    phi_33769_ = (1f <= _e9546_);
                                                }
                                                let _e9565_ = phi_33769_;
                                                let _e9566_ = select(_e9546_, 1f, _e9565_);
                                                if (_e9551_ != _e9551_) {
                                                    phi_33784_ = true;
                                                } else {
                                                    phi_33784_ = (1f <= _e9551_);
                                                }
                                                let _e9570_ = phi_33784_;
                                                let _e9571_ = select(_e9551_, 1f, _e9570_);
                                                if (_e9556_ != _e9556_) {
                                                    phi_33799_ = true;
                                                } else {
                                                    phi_33799_ = (1f <= _e9556_);
                                                }
                                                let _e9575_ = phi_33799_;
                                                let _e9576_ = select(_e9556_, 1f, _e9575_);
                                                let _e9581_ = (0.5f * (_e9561_ + _e9521_));
                                                let _e9582_ = (0.5f * (_e9566_ + _e9526_));
                                                let _e9583_ = (0.5f * (_e9571_ + _e9531_));
                                                let _e9584_ = (0.5f * (_e9576_ + _e9536_));
                                                let _e9585_ = (_e9561_ - _e9521_);
                                                let _e9586_ = (_e9566_ - _e9526_);
                                                let _e9587_ = (_e9571_ - _e9531_);
                                                let _e9588_ = (_e9576_ - _e9536_);
                                                let _e9589_ = (0.5f * _e9585_);
                                                let _e9590_ = (0.5f * _e9586_);
                                                let _e9591_ = (0.5f * _e9587_);
                                                let _e9592_ = (0.5f * _e9588_);
                                                let _e9593_ = (_e9581_ * _e9581_);
                                                let _e9594_ = (_e9582_ * _e9582_);
                                                let _e9595_ = (_e9583_ * _e9583_);
                                                let _e9596_ = (_e9584_ * _e9584_);
                                                let _e9597_ = (_e9589_ * _e9589_);
                                                let _e9598_ = (_e9590_ * _e9590_);
                                                let _e9599_ = (_e9591_ * _e9591_);
                                                let _e9600_ = (_e9592_ * _e9592_);
                                                if (_e9585_ != _e9585_) {
                                                    phi_33814_ = true;
                                                } else {
                                                    phi_33814_ = (0f >= _e9585_);
                                                }
                                                let _e9604_ = phi_33814_;
                                                if (_e9586_ != _e9586_) {
                                                    phi_33829_ = true;
                                                } else {
                                                    phi_33829_ = (0f >= _e9586_);
                                                }
                                                let _e9609_ = phi_33829_;
                                                if (_e9587_ != _e9587_) {
                                                    phi_33844_ = true;
                                                } else {
                                                    phi_33844_ = (0f >= _e9587_);
                                                }
                                                let _e9614_ = phi_33844_;
                                                if (_e9588_ != _e9588_) {
                                                    phi_33859_ = true;
                                                } else {
                                                    phi_33859_ = (0f >= _e9588_);
                                                }
                                                let _e9619_ = phi_33859_;
                                                let _e9621_ = (_e9484_ - _e9465_);
                                                let _e9622_ = (_e9488_ - _e9470_);
                                                let _e9623_ = (_e9492_ - _e9475_);
                                                let _e9624_ = (_e9496_ - _e9480_);
                                                let _e9627_ = bitcast<f32>((bitcast<u32>(_e9397_) & 2147483647u));
                                                let _e9630_ = bitcast<f32>((bitcast<u32>(_e9402_) & 2147483647u));
                                                let _e9633_ = bitcast<f32>((bitcast<u32>(_e9407_) & 2147483647u));
                                                let _e9636_ = bitcast<f32>((bitcast<u32>(_e9412_) & 2147483647u));
                                                let _e9639_ = bitcast<f32>((bitcast<u32>(_e9445_) & 2147483647u));
                                                let _e9642_ = bitcast<f32>((bitcast<u32>(_e9450_) & 2147483647u));
                                                let _e9645_ = bitcast<f32>((bitcast<u32>(_e9455_) & 2147483647u));
                                                let _e9648_ = bitcast<f32>((bitcast<u32>(_e9460_) & 2147483647u));
                                                if (_e9627_ != _e9627_) {
                                                    phi_34114_ = true;
                                                } else {
                                                    phi_34114_ = (_e9639_ >= _e9627_);
                                                }
                                                let _e9652_ = phi_34114_;
                                                let _e9653_ = select(_e9627_, _e9639_, _e9652_);
                                                if (_e9630_ != _e9630_) {
                                                    phi_34129_ = true;
                                                } else {
                                                    phi_34129_ = (_e9642_ >= _e9630_);
                                                }
                                                let _e9657_ = phi_34129_;
                                                let _e9658_ = select(_e9630_, _e9642_, _e9657_);
                                                if (_e9633_ != _e9633_) {
                                                    phi_34144_ = true;
                                                } else {
                                                    phi_34144_ = (_e9645_ >= _e9633_);
                                                }
                                                let _e9662_ = phi_34144_;
                                                let _e9663_ = select(_e9633_, _e9645_, _e9662_);
                                                if (_e9636_ != _e9636_) {
                                                    phi_34159_ = true;
                                                } else {
                                                    phi_34159_ = (_e9648_ >= _e9636_);
                                                }
                                                let _e9667_ = phi_34159_;
                                                let _e9668_ = select(_e9636_, _e9648_, _e9667_);
                                                let _e9673_ = (1f - (_e9593_ + _e9597_));
                                                let _e9674_ = (1f - (_e9594_ + _e9598_));
                                                let _e9675_ = (1f - (_e9595_ + _e9599_));
                                                let _e9676_ = (1f - (_e9596_ + _e9600_));
                                                let _e9702_ = select(((_e9581_ * _e9673_) * ((_e9673_ * _e9673_) + ((4f * _e9593_) * _e9597_))), 0f, (_e9653_ <= 0f));
                                                let _e9704_ = select(((_e9582_ * _e9674_) * ((_e9674_ * _e9674_) + ((4f * _e9594_) * _e9598_))), 0f, (_e9658_ <= 0f));
                                                let _e9706_ = select(((_e9583_ * _e9675_) * ((_e9675_ * _e9675_) + ((4f * _e9595_) * _e9599_))), 0f, (_e9663_ <= 0f));
                                                let _e9708_ = select(((_e9584_ * _e9676_) * ((_e9676_ * _e9676_) + ((4f * _e9596_) * _e9600_))), 0f, (_e9668_ <= 0f));
                                                if (_e9653_ <= 1f) {
                                                    if (_e9658_ <= 1f) {
                                                        let _e9712_ = (_e9663_ <= 1f);
                                                        if _e9712_ {
                                                            phi_21739_ = (_e9668_ <= 1f);
                                                        } else {
                                                            phi_21739_ = bool();
                                                        }
                                                        let _e9715_ = phi_21739_;
                                                        phi_21741_ = _e9715_;
                                                        phi_21742_ = select(true, false, _e9712_);
                                                    } else {
                                                        phi_21741_ = bool();
                                                        phi_21742_ = true;
                                                    }
                                                    let _e9718_ = phi_21741_;
                                                    let _e9720_ = phi_21742_;
                                                    phi_21743_ = _e9718_;
                                                    phi_21744_ = _e9720_;
                                                } else {
                                                    phi_21743_ = bool();
                                                    phi_21744_ = true;
                                                }
                                                let _e9722_ = phi_21743_;
                                                let _e9724_ = phi_21744_;
                                                if select(_e9722_, false, _e9724_) {
                                                    phi_21981_ = type_32_(array<f32, 4>(_e9702_, _e9704_, _e9706_, _e9708_));
                                                } else {
                                                    if (_e9621_ != _e9621_) {
                                                        phi_34174_ = true;
                                                    } else {
                                                        phi_34174_ = (0.000001f >= _e9621_);
                                                    }
                                                    let _e9729_ = phi_34174_;
                                                    let _e9730_ = select(_e9621_, 0.000001f, _e9729_);
                                                    if (_e9622_ != _e9622_) {
                                                        phi_34189_ = true;
                                                    } else {
                                                        phi_34189_ = (0.000001f >= _e9622_);
                                                    }
                                                    let _e9734_ = phi_34189_;
                                                    let _e9735_ = select(_e9622_, 0.000001f, _e9734_);
                                                    if (_e9623_ != _e9623_) {
                                                        phi_34204_ = true;
                                                    } else {
                                                        phi_34204_ = (0.000001f >= _e9623_);
                                                    }
                                                    let _e9739_ = phi_34204_;
                                                    let _e9740_ = select(_e9623_, 0.000001f, _e9739_);
                                                    if (_e9624_ != _e9624_) {
                                                        phi_34219_ = true;
                                                    } else {
                                                        phi_34219_ = (0.000001f >= _e9624_);
                                                    }
                                                    let _e9744_ = phi_34219_;
                                                    let _e9745_ = select(_e9624_, 0.000001f, _e9744_);
                                                    let _e9748_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9730_)));
                                                    let _e9751_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9735_)));
                                                    let _e9754_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9740_)));
                                                    let _e9757_ = bitcast<f32>((2129859011u - bitcast<u32>(_e9745_)));
                                                    let _e9767_ = (_e9748_ + (_e9748_ * (1f - (_e9730_ * _e9748_))));
                                                    let _e9769_ = (_e9751_ + (_e9751_ * (1f - (_e9735_ * _e9751_))));
                                                    let _e9771_ = (_e9754_ + (_e9754_ * (1f - (_e9740_ * _e9754_))));
                                                    let _e9773_ = (_e9757_ + (_e9757_ * (1f - (_e9745_ * _e9757_))));
                                                    phi_21981_ = type_32_(array<f32, 4>(select(select(select(_e9702_, ((select(_e9585_, 0f, _e9604_) * _e9702_) * (_e9767_ + (_e9767_ * (1f - (_e9730_ * _e9767_))))), (select((select(0f, 1f, (-1f <= _e9465_)) * select(0f, 1f, (_e9484_ <= 1f))), 1f, (_e9621_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e9465_)), 0f, (_e9484_ <= -1f)), select(select(select(_e9704_, ((select(_e9586_, 0f, _e9609_) * _e9704_) * (_e9769_ + (_e9769_ * (1f - (_e9735_ * _e9769_))))), (select((select(0f, 1f, (-1f <= _e9470_)) * select(0f, 1f, (_e9488_ <= 1f))), 1f, (_e9622_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e9470_)), 0f, (_e9488_ <= -1f)), select(select(select(_e9706_, ((select(_e9587_, 0f, _e9614_) * _e9706_) * (_e9771_ + (_e9771_ * (1f - (_e9740_ * _e9771_))))), (select((select(0f, 1f, (-1f <= _e9475_)) * select(0f, 1f, (_e9492_ <= 1f))), 1f, (_e9623_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e9475_)), 0f, (_e9492_ <= -1f)), select(select(select(_e9708_, ((select(_e9588_, 0f, _e9619_) * _e9708_) * (_e9773_ + (_e9773_ * (1f - (_e9745_ * _e9773_))))), (select((select(0f, 1f, (-1f <= _e9480_)) * select(0f, 1f, (_e9496_ <= 1f))), 1f, (_e9624_ <= 0.000001f)) <= 0f)), 0f, (1f <= _e9480_)), 0f, (_e9496_ <= -1f))));
                                                }
                                                let _e9854_ = phi_21981_;
                                                phi_22017_ = type_33_(type_32_(array<f32, 4>(((_e9248_ * (1f - _e8538_)) + (_e8538_ * _e9854_.member[0])), ((_e9250_ * (1f - _e8543_)) + (_e8543_ * _e9854_.member[1])), ((_e9252_ * (1f - _e8548_)) + (_e8548_ * _e9854_.member[2])), ((_e9254_ * (1f - _e8553_)) + (_e8553_ * _e9854_.member[3])))), type_32_(_e9300_), type_32_(_e8682_));
                                            }
                                            let _e9889_ = phi_22017_;
                                            let _e9906_ = select(0f, _e479_.x, (bitcast<f32>((bitcast<u32>(_e479_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9908_ = select(0f, _e479_.y, (bitcast<f32>((bitcast<u32>(_e479_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9910_ = select(0f, _e479_.z, (bitcast<f32>((bitcast<u32>(_e479_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9912_ = select(0f, _e479_.w, (bitcast<f32>((bitcast<u32>(_e479_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9913_ = (_e9906_ != _e9906_);
                                            if _e9913_ {
                                                phi_34234_ = true;
                                            } else {
                                                phi_34234_ = (0f >= _e9906_);
                                            }
                                            let _e9916_ = phi_34234_;
                                            let _e9917_ = select(_e9906_, 0f, _e9916_);
                                            let _e9918_ = (_e9908_ != _e9908_);
                                            if _e9918_ {
                                                phi_34249_ = true;
                                            } else {
                                                phi_34249_ = (0f >= _e9908_);
                                            }
                                            let _e9921_ = phi_34249_;
                                            let _e9922_ = select(_e9908_, 0f, _e9921_);
                                            let _e9923_ = (_e9910_ != _e9910_);
                                            if _e9923_ {
                                                phi_34264_ = true;
                                            } else {
                                                phi_34264_ = (0f >= _e9910_);
                                            }
                                            let _e9926_ = phi_34264_;
                                            let _e9927_ = select(_e9910_, 0f, _e9926_);
                                            let _e9928_ = (_e9912_ != _e9912_);
                                            if _e9928_ {
                                                phi_34279_ = true;
                                            } else {
                                                phi_34279_ = (0f >= _e9912_);
                                            }
                                            let _e9931_ = phi_34279_;
                                            let _e9932_ = select(_e9912_, 0f, _e9931_);
                                            if (_e9917_ != _e9917_) {
                                                phi_34294_ = true;
                                            } else {
                                                phi_34294_ = (1000000000000f <= _e9917_);
                                            }
                                            let _e9936_ = phi_34294_;
                                            if (_e9922_ != _e9922_) {
                                                phi_34309_ = true;
                                            } else {
                                                phi_34309_ = (1000000000000f <= _e9922_);
                                            }
                                            let _e9941_ = phi_34309_;
                                            if (_e9927_ != _e9927_) {
                                                phi_34324_ = true;
                                            } else {
                                                phi_34324_ = (1000000000000f <= _e9927_);
                                            }
                                            let _e9946_ = phi_34324_;
                                            if (_e9932_ != _e9932_) {
                                                phi_34339_ = true;
                                            } else {
                                                phi_34339_ = (1000000000000f <= _e9932_);
                                            }
                                            let _e9951_ = phi_34339_;
                                            let _e9966_ = select(0f, _e486_.x, (bitcast<f32>((bitcast<u32>(_e486_.x) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9968_ = select(0f, _e486_.y, (bitcast<f32>((bitcast<u32>(_e486_.y) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9970_ = select(0f, _e486_.z, (bitcast<f32>((bitcast<u32>(_e486_.z) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9972_ = select(0f, _e486_.w, (bitcast<f32>((bitcast<u32>(_e486_.w) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e9973_ = (_e9966_ != _e9966_);
                                            if _e9973_ {
                                                phi_34354_ = true;
                                            } else {
                                                phi_34354_ = (0f >= _e9966_);
                                            }
                                            let _e9976_ = phi_34354_;
                                            let _e9977_ = select(_e9966_, 0f, _e9976_);
                                            let _e9978_ = (_e9968_ != _e9968_);
                                            if _e9978_ {
                                                phi_34369_ = true;
                                            } else {
                                                phi_34369_ = (0f >= _e9968_);
                                            }
                                            let _e9981_ = phi_34369_;
                                            let _e9982_ = select(_e9968_, 0f, _e9981_);
                                            let _e9983_ = (_e9970_ != _e9970_);
                                            if _e9983_ {
                                                phi_34384_ = true;
                                            } else {
                                                phi_34384_ = (0f >= _e9970_);
                                            }
                                            let _e9986_ = phi_34384_;
                                            let _e9987_ = select(_e9970_, 0f, _e9986_);
                                            let _e9988_ = (_e9972_ != _e9972_);
                                            if _e9988_ {
                                                phi_34399_ = true;
                                            } else {
                                                phi_34399_ = (0f >= _e9972_);
                                            }
                                            let _e9991_ = phi_34399_;
                                            let _e9992_ = select(_e9972_, 0f, _e9991_);
                                            if (_e9977_ != _e9977_) {
                                                phi_34414_ = true;
                                            } else {
                                                phi_34414_ = (1000000000000f <= _e9977_);
                                            }
                                            let _e9996_ = phi_34414_;
                                            if (_e9982_ != _e9982_) {
                                                phi_34429_ = true;
                                            } else {
                                                phi_34429_ = (1000000000000f <= _e9982_);
                                            }
                                            let _e10001_ = phi_34429_;
                                            if (_e9987_ != _e9987_) {
                                                phi_34444_ = true;
                                            } else {
                                                phi_34444_ = (1000000000000f <= _e9987_);
                                            }
                                            let _e10006_ = phi_34444_;
                                            if (_e9992_ != _e9992_) {
                                                phi_34459_ = true;
                                            } else {
                                                phi_34459_ = (1000000000000f <= _e9992_);
                                            }
                                            let _e10011_ = phi_34459_;
                                            let _e10015_ = bitcast<u32>(_e7032_.member[0]);
                                            let _e10016_ = (_e10015_ & 2139095040u);
                                            if (_e10016_ == 2139095040u) {
                                                phi_22122_ = true;
                                            } else {
                                                phi_22122_ = (_e10016_ == 0u);
                                            }
                                            let _e10020_ = phi_22122_;
                                            let _e10025_ = bitcast<u32>(_e7032_.member[1]);
                                            let _e10026_ = (_e10025_ & 2139095040u);
                                            if (_e10026_ == 2139095040u) {
                                                phi_22139_ = true;
                                            } else {
                                                phi_22139_ = (_e10026_ == 0u);
                                            }
                                            let _e10030_ = phi_22139_;
                                            let _e10035_ = bitcast<u32>(_e7032_.member[2]);
                                            let _e10036_ = (_e10035_ & 2139095040u);
                                            if (_e10036_ == 2139095040u) {
                                                phi_22156_ = true;
                                            } else {
                                                phi_22156_ = (_e10036_ == 0u);
                                            }
                                            let _e10040_ = phi_22156_;
                                            let _e10045_ = bitcast<u32>(_e7032_.member[3]);
                                            let _e10046_ = (_e10045_ & 2139095040u);
                                            if (_e10046_ == 2139095040u) {
                                                phi_22173_ = true;
                                            } else {
                                                phi_22173_ = (_e10046_ == 0u);
                                            }
                                            let _e10050_ = phi_22173_;
                                            let _e10055_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10015_, 0u, _e10020_))) & 2147483647u));
                                            let _e10058_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10025_, 0u, _e10030_))) & 2147483647u));
                                            let _e10061_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10035_, 0u, _e10040_))) & 2147483647u));
                                            let _e10064_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10045_, 0u, _e10050_))) & 2147483647u));
                                            if (_e10055_ != _e10055_) {
                                                phi_34474_ = true;
                                            } else {
                                                phi_34474_ = (1000000f <= _e10055_);
                                            }
                                            let _e10068_ = phi_34474_;
                                            if (_e10058_ != _e10058_) {
                                                phi_34489_ = true;
                                            } else {
                                                phi_34489_ = (1000000f <= _e10058_);
                                            }
                                            let _e10073_ = phi_34489_;
                                            if (_e10061_ != _e10061_) {
                                                phi_34504_ = true;
                                            } else {
                                                phi_34504_ = (1000000f <= _e10061_);
                                            }
                                            let _e10078_ = phi_34504_;
                                            if (_e10064_ != _e10064_) {
                                                phi_34519_ = true;
                                            } else {
                                                phi_34519_ = (1000000f <= _e10064_);
                                            }
                                            let _e10083_ = phi_34519_;
                                            let _e10091_ = (((select(_e9917_, 1000000000000f, _e9936_) * select(_e10055_, 1000000f, _e10068_)) + (select(_e9922_, 1000000000000f, _e9941_) * select(_e10058_, 1000000f, _e10073_))) + ((select(_e9927_, 1000000000000f, _e9946_) * select(_e10061_, 1000000f, _e10078_)) + (select(_e9932_, 1000000000000f, _e9951_) * select(_e10064_, 1000000f, _e10083_))));
                                            let _e10092_ = (select(_e9977_, 1000000000000f, _e9996_) * _e10091_);
                                            let _e10093_ = (select(_e9982_, 1000000000000f, _e10001_) * _e10091_);
                                            let _e10094_ = (select(_e9987_, 1000000000000f, _e10006_) * _e10091_);
                                            let _e10095_ = (select(_e9992_, 1000000000000f, _e10011_) * _e10091_);
                                            if (_e10092_ != _e10092_) {
                                                phi_34534_ = true;
                                            } else {
                                                phi_34534_ = (1000000000000f <= _e10092_);
                                            }
                                            let _e10099_ = phi_34534_;
                                            let _e10100_ = select(_e10092_, 1000000000000f, _e10099_);
                                            if (_e10093_ != _e10093_) {
                                                phi_34549_ = true;
                                            } else {
                                                phi_34549_ = (1000000000000f <= _e10093_);
                                            }
                                            let _e10104_ = phi_34549_;
                                            let _e10105_ = select(_e10093_, 1000000000000f, _e10104_);
                                            if (_e10094_ != _e10094_) {
                                                phi_34564_ = true;
                                            } else {
                                                phi_34564_ = (1000000000000f <= _e10094_);
                                            }
                                            let _e10109_ = phi_34564_;
                                            let _e10110_ = select(_e10094_, 1000000000000f, _e10109_);
                                            if (_e10095_ != _e10095_) {
                                                phi_34579_ = true;
                                            } else {
                                                phi_34579_ = (1000000000000f <= _e10095_);
                                            }
                                            let _e10114_ = phi_34579_;
                                            let _e10115_ = select(_e10095_, 1000000000000f, _e10114_);
                                            let _e10116_ = (1f + _e10100_);
                                            let _e10117_ = (1f + _e10105_);
                                            let _e10118_ = (1f + _e10110_);
                                            let _e10119_ = (1f + _e10115_);
                                            let _e10122_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10116_)));
                                            let _e10125_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10117_)));
                                            let _e10128_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10118_)));
                                            let _e10131_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10119_)));
                                            let _e10141_ = (_e10122_ + (_e10122_ * (1f - (_e10116_ * _e10122_))));
                                            let _e10143_ = (_e10125_ + (_e10125_ * (1f - (_e10117_ * _e10125_))));
                                            let _e10145_ = (_e10128_ + (_e10128_ * (1f - (_e10118_ * _e10128_))));
                                            let _e10147_ = (_e10131_ + (_e10131_ * (1f - (_e10119_ * _e10131_))));
                                            let _e10157_ = (_e10141_ + (_e10141_ * (1f - (_e10116_ * _e10141_))));
                                            let _e10159_ = (_e10143_ + (_e10143_ * (1f - (_e10117_ * _e10143_))));
                                            let _e10161_ = (_e10145_ + (_e10145_ * (1f - (_e10118_ * _e10145_))));
                                            let _e10163_ = (_e10147_ + (_e10147_ * (1f - (_e10119_ * _e10147_))));
                                            if (_e10157_ != _e10157_) {
                                                phi_34594_ = true;
                                            } else {
                                                phi_34594_ = (0f >= _e10157_);
                                            }
                                            let _e10167_ = phi_34594_;
                                            let _e10168_ = select(_e10157_, 0f, _e10167_);
                                            if (_e10159_ != _e10159_) {
                                                phi_34609_ = true;
                                            } else {
                                                phi_34609_ = (0f >= _e10159_);
                                            }
                                            let _e10172_ = phi_34609_;
                                            let _e10173_ = select(_e10159_, 0f, _e10172_);
                                            if (_e10161_ != _e10161_) {
                                                phi_34624_ = true;
                                            } else {
                                                phi_34624_ = (0f >= _e10161_);
                                            }
                                            let _e10177_ = phi_34624_;
                                            let _e10178_ = select(_e10161_, 0f, _e10177_);
                                            if (_e10163_ != _e10163_) {
                                                phi_34639_ = true;
                                            } else {
                                                phi_34639_ = (0f >= _e10163_);
                                            }
                                            let _e10182_ = phi_34639_;
                                            let _e10183_ = select(_e10163_, 0f, _e10182_);
                                            if (_e10168_ != _e10168_) {
                                                phi_34654_ = true;
                                            } else {
                                                phi_34654_ = (1f <= _e10168_);
                                            }
                                            let _e10187_ = phi_34654_;
                                            if (_e10173_ != _e10173_) {
                                                phi_34669_ = true;
                                            } else {
                                                phi_34669_ = (1f <= _e10173_);
                                            }
                                            let _e10192_ = phi_34669_;
                                            if (_e10178_ != _e10178_) {
                                                phi_34684_ = true;
                                            } else {
                                                phi_34684_ = (1f <= _e10178_);
                                            }
                                            let _e10197_ = phi_34684_;
                                            if (_e10183_ != _e10183_) {
                                                phi_34699_ = true;
                                            } else {
                                                phi_34699_ = (1f <= _e10183_);
                                            }
                                            let _e10202_ = phi_34699_;
                                            let _e10205_ = select(select(_e10168_, 1f, _e10187_), 1f, (_e10100_ <= 0f));
                                            let _e10207_ = select(select(_e10173_, 1f, _e10192_), 1f, (_e10105_ <= 0f));
                                            let _e10209_ = select(select(_e10178_, 1f, _e10197_), 1f, (_e10110_ <= 0f));
                                            let _e10211_ = select(select(_e10183_, 1f, _e10202_), 1f, (_e10115_ <= 0f));
                                            if _e9913_ {
                                                phi_34714_ = true;
                                            } else {
                                                phi_34714_ = (0f >= _e9906_);
                                            }
                                            let _e10215_ = phi_34714_;
                                            let _e10216_ = select(_e9906_, 0f, _e10215_);
                                            if _e9918_ {
                                                phi_34729_ = true;
                                            } else {
                                                phi_34729_ = (0f >= _e9908_);
                                            }
                                            let _e10219_ = phi_34729_;
                                            let _e10220_ = select(_e9908_, 0f, _e10219_);
                                            if _e9923_ {
                                                phi_34744_ = true;
                                            } else {
                                                phi_34744_ = (0f >= _e9910_);
                                            }
                                            let _e10223_ = phi_34744_;
                                            let _e10224_ = select(_e9910_, 0f, _e10223_);
                                            if _e9928_ {
                                                phi_34759_ = true;
                                            } else {
                                                phi_34759_ = (0f >= _e9912_);
                                            }
                                            let _e10227_ = phi_34759_;
                                            let _e10228_ = select(_e9912_, 0f, _e10227_);
                                            if (_e10216_ != _e10216_) {
                                                phi_34774_ = true;
                                            } else {
                                                phi_34774_ = (1000000000000f <= _e10216_);
                                            }
                                            let _e10232_ = phi_34774_;
                                            if (_e10220_ != _e10220_) {
                                                phi_34789_ = true;
                                            } else {
                                                phi_34789_ = (1000000000000f <= _e10220_);
                                            }
                                            let _e10237_ = phi_34789_;
                                            if (_e10224_ != _e10224_) {
                                                phi_34804_ = true;
                                            } else {
                                                phi_34804_ = (1000000000000f <= _e10224_);
                                            }
                                            let _e10242_ = phi_34804_;
                                            if (_e10228_ != _e10228_) {
                                                phi_34819_ = true;
                                            } else {
                                                phi_34819_ = (1000000000000f <= _e10228_);
                                            }
                                            let _e10247_ = phi_34819_;
                                            if _e9973_ {
                                                phi_34834_ = true;
                                            } else {
                                                phi_34834_ = (0f >= _e9966_);
                                            }
                                            let _e10251_ = phi_34834_;
                                            let _e10252_ = select(_e9966_, 0f, _e10251_);
                                            if _e9978_ {
                                                phi_34849_ = true;
                                            } else {
                                                phi_34849_ = (0f >= _e9968_);
                                            }
                                            let _e10255_ = phi_34849_;
                                            let _e10256_ = select(_e9968_, 0f, _e10255_);
                                            if _e9983_ {
                                                phi_34864_ = true;
                                            } else {
                                                phi_34864_ = (0f >= _e9970_);
                                            }
                                            let _e10259_ = phi_34864_;
                                            let _e10260_ = select(_e9970_, 0f, _e10259_);
                                            if _e9988_ {
                                                phi_34879_ = true;
                                            } else {
                                                phi_34879_ = (0f >= _e9972_);
                                            }
                                            let _e10263_ = phi_34879_;
                                            let _e10264_ = select(_e9972_, 0f, _e10263_);
                                            if (_e10252_ != _e10252_) {
                                                phi_34894_ = true;
                                            } else {
                                                phi_34894_ = (1000000000000f <= _e10252_);
                                            }
                                            let _e10268_ = phi_34894_;
                                            if (_e10256_ != _e10256_) {
                                                phi_34909_ = true;
                                            } else {
                                                phi_34909_ = (1000000000000f <= _e10256_);
                                            }
                                            let _e10273_ = phi_34909_;
                                            if (_e10260_ != _e10260_) {
                                                phi_34924_ = true;
                                            } else {
                                                phi_34924_ = (1000000000000f <= _e10260_);
                                            }
                                            let _e10278_ = phi_34924_;
                                            if (_e10264_ != _e10264_) {
                                                phi_34939_ = true;
                                            } else {
                                                phi_34939_ = (1000000000000f <= _e10264_);
                                            }
                                            let _e10283_ = phi_34939_;
                                            let _e10287_ = bitcast<u32>(_e7030_.member[0]);
                                            let _e10288_ = (_e10287_ & 2139095040u);
                                            if (_e10288_ == 2139095040u) {
                                                phi_22392_ = true;
                                            } else {
                                                phi_22392_ = (_e10288_ == 0u);
                                            }
                                            let _e10292_ = phi_22392_;
                                            let _e10297_ = bitcast<u32>(_e7030_.member[1]);
                                            let _e10298_ = (_e10297_ & 2139095040u);
                                            if (_e10298_ == 2139095040u) {
                                                phi_22409_ = true;
                                            } else {
                                                phi_22409_ = (_e10298_ == 0u);
                                            }
                                            let _e10302_ = phi_22409_;
                                            let _e10307_ = bitcast<u32>(_e7030_.member[2]);
                                            let _e10308_ = (_e10307_ & 2139095040u);
                                            if (_e10308_ == 2139095040u) {
                                                phi_22426_ = true;
                                            } else {
                                                phi_22426_ = (_e10308_ == 0u);
                                            }
                                            let _e10312_ = phi_22426_;
                                            let _e10317_ = bitcast<u32>(_e7030_.member[3]);
                                            let _e10318_ = (_e10317_ & 2139095040u);
                                            if (_e10318_ == 2139095040u) {
                                                phi_22443_ = true;
                                            } else {
                                                phi_22443_ = (_e10318_ == 0u);
                                            }
                                            let _e10322_ = phi_22443_;
                                            let _e10327_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10287_, 0u, _e10292_))) & 2147483647u));
                                            let _e10330_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10297_, 0u, _e10302_))) & 2147483647u));
                                            let _e10333_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10307_, 0u, _e10312_))) & 2147483647u));
                                            let _e10336_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e10317_, 0u, _e10322_))) & 2147483647u));
                                            if (_e10327_ != _e10327_) {
                                                phi_34954_ = true;
                                            } else {
                                                phi_34954_ = (1000000f <= _e10327_);
                                            }
                                            let _e10340_ = phi_34954_;
                                            if (_e10330_ != _e10330_) {
                                                phi_34969_ = true;
                                            } else {
                                                phi_34969_ = (1000000f <= _e10330_);
                                            }
                                            let _e10345_ = phi_34969_;
                                            if (_e10333_ != _e10333_) {
                                                phi_34984_ = true;
                                            } else {
                                                phi_34984_ = (1000000f <= _e10333_);
                                            }
                                            let _e10350_ = phi_34984_;
                                            if (_e10336_ != _e10336_) {
                                                phi_34999_ = true;
                                            } else {
                                                phi_34999_ = (1000000f <= _e10336_);
                                            }
                                            let _e10355_ = phi_34999_;
                                            let _e10363_ = (((select(_e10216_, 1000000000000f, _e10232_) * select(_e10327_, 1000000f, _e10340_)) + (select(_e10220_, 1000000000000f, _e10237_) * select(_e10330_, 1000000f, _e10345_))) + ((select(_e10224_, 1000000000000f, _e10242_) * select(_e10333_, 1000000f, _e10350_)) + (select(_e10228_, 1000000000000f, _e10247_) * select(_e10336_, 1000000f, _e10355_))));
                                            let _e10364_ = (select(_e10252_, 1000000000000f, _e10268_) * _e10363_);
                                            let _e10365_ = (select(_e10256_, 1000000000000f, _e10273_) * _e10363_);
                                            let _e10366_ = (select(_e10260_, 1000000000000f, _e10278_) * _e10363_);
                                            let _e10367_ = (select(_e10264_, 1000000000000f, _e10283_) * _e10363_);
                                            if (_e10364_ != _e10364_) {
                                                phi_35014_ = true;
                                            } else {
                                                phi_35014_ = (1000000000000f <= _e10364_);
                                            }
                                            let _e10371_ = phi_35014_;
                                            let _e10372_ = select(_e10364_, 1000000000000f, _e10371_);
                                            if (_e10365_ != _e10365_) {
                                                phi_35029_ = true;
                                            } else {
                                                phi_35029_ = (1000000000000f <= _e10365_);
                                            }
                                            let _e10376_ = phi_35029_;
                                            let _e10377_ = select(_e10365_, 1000000000000f, _e10376_);
                                            if (_e10366_ != _e10366_) {
                                                phi_35044_ = true;
                                            } else {
                                                phi_35044_ = (1000000000000f <= _e10366_);
                                            }
                                            let _e10381_ = phi_35044_;
                                            let _e10382_ = select(_e10366_, 1000000000000f, _e10381_);
                                            if (_e10367_ != _e10367_) {
                                                phi_35059_ = true;
                                            } else {
                                                phi_35059_ = (1000000000000f <= _e10367_);
                                            }
                                            let _e10386_ = phi_35059_;
                                            let _e10387_ = select(_e10367_, 1000000000000f, _e10386_);
                                            let _e10388_ = (1f + _e10372_);
                                            let _e10389_ = (1f + _e10377_);
                                            let _e10390_ = (1f + _e10382_);
                                            let _e10391_ = (1f + _e10387_);
                                            let _e10394_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10388_)));
                                            let _e10397_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10389_)));
                                            let _e10400_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10390_)));
                                            let _e10403_ = bitcast<f32>((2129859011u - bitcast<u32>(_e10391_)));
                                            let _e10413_ = (_e10394_ + (_e10394_ * (1f - (_e10388_ * _e10394_))));
                                            let _e10415_ = (_e10397_ + (_e10397_ * (1f - (_e10389_ * _e10397_))));
                                            let _e10417_ = (_e10400_ + (_e10400_ * (1f - (_e10390_ * _e10400_))));
                                            let _e10419_ = (_e10403_ + (_e10403_ * (1f - (_e10391_ * _e10403_))));
                                            let _e10429_ = (_e10413_ + (_e10413_ * (1f - (_e10388_ * _e10413_))));
                                            let _e10431_ = (_e10415_ + (_e10415_ * (1f - (_e10389_ * _e10415_))));
                                            let _e10433_ = (_e10417_ + (_e10417_ * (1f - (_e10390_ * _e10417_))));
                                            let _e10435_ = (_e10419_ + (_e10419_ * (1f - (_e10391_ * _e10419_))));
                                            if (_e10429_ != _e10429_) {
                                                phi_35074_ = true;
                                            } else {
                                                phi_35074_ = (0f >= _e10429_);
                                            }
                                            let _e10439_ = phi_35074_;
                                            let _e10440_ = select(_e10429_, 0f, _e10439_);
                                            if (_e10431_ != _e10431_) {
                                                phi_35089_ = true;
                                            } else {
                                                phi_35089_ = (0f >= _e10431_);
                                            }
                                            let _e10444_ = phi_35089_;
                                            let _e10445_ = select(_e10431_, 0f, _e10444_);
                                            if (_e10433_ != _e10433_) {
                                                phi_35104_ = true;
                                            } else {
                                                phi_35104_ = (0f >= _e10433_);
                                            }
                                            let _e10449_ = phi_35104_;
                                            let _e10450_ = select(_e10433_, 0f, _e10449_);
                                            if (_e10435_ != _e10435_) {
                                                phi_35119_ = true;
                                            } else {
                                                phi_35119_ = (0f >= _e10435_);
                                            }
                                            let _e10454_ = phi_35119_;
                                            let _e10455_ = select(_e10435_, 0f, _e10454_);
                                            if (_e10440_ != _e10440_) {
                                                phi_35134_ = true;
                                            } else {
                                                phi_35134_ = (1f <= _e10440_);
                                            }
                                            let _e10459_ = phi_35134_;
                                            if (_e10445_ != _e10445_) {
                                                phi_35149_ = true;
                                            } else {
                                                phi_35149_ = (1f <= _e10445_);
                                            }
                                            let _e10464_ = phi_35149_;
                                            if (_e10450_ != _e10450_) {
                                                phi_35164_ = true;
                                            } else {
                                                phi_35164_ = (1f <= _e10450_);
                                            }
                                            let _e10469_ = phi_35164_;
                                            if (_e10455_ != _e10455_) {
                                                phi_35179_ = true;
                                            } else {
                                                phi_35179_ = (1f <= _e10455_);
                                            }
                                            let _e10474_ = phi_35179_;
                                            let _e10477_ = select(select(_e10440_, 1f, _e10459_), 1f, (_e10372_ <= 0f));
                                            let _e10479_ = select(select(_e10445_, 1f, _e10464_), 1f, (_e10377_ <= 0f));
                                            let _e10481_ = select(select(_e10450_, 1f, _e10469_), 1f, (_e10382_ <= 0f));
                                            let _e10483_ = select(select(_e10455_, 1f, _e10474_), 1f, (_e10387_ <= 0f));
                                            let _e10574_ = bitcast<u32>(((_e1006_ * _e10205_) * ((_e1061_ * _e8510_.member.member[0]) + (_e790_ * (_e7024_.member[0] * _e7028_.member[0])))));
                                            let _e10575_ = (_e10574_ & 2139095040u);
                                            if (_e10575_ == 2139095040u) {
                                                phi_22645_ = true;
                                            } else {
                                                phi_22645_ = (_e10575_ == 0u);
                                            }
                                            let _e10579_ = phi_22645_;
                                            let _e10582_ = bitcast<u32>(((_e1024_ * _e10207_) * ((_e1062_ * _e8510_.member.member[1]) + (_e808_ * (_e7024_.member[1] * _e7028_.member[1])))));
                                            let _e10583_ = (_e10582_ & 2139095040u);
                                            if (_e10583_ == 2139095040u) {
                                                phi_22661_ = true;
                                            } else {
                                                phi_22661_ = (_e10583_ == 0u);
                                            }
                                            let _e10587_ = phi_22661_;
                                            let _e10590_ = bitcast<u32>(((_e1042_ * _e10209_) * ((_e1063_ * _e8510_.member.member[2]) + (_e826_ * (_e7024_.member[2] * _e7028_.member[2])))));
                                            let _e10591_ = (_e10590_ & 2139095040u);
                                            if (_e10591_ == 2139095040u) {
                                                phi_22677_ = true;
                                            } else {
                                                phi_22677_ = (_e10591_ == 0u);
                                            }
                                            let _e10595_ = phi_22677_;
                                            let _e10598_ = bitcast<u32>(((_e1060_ * _e10211_) * ((_e1064_ * _e8510_.member.member[3]) + (_e844_ * (_e7024_.member[3] * _e7028_.member[3])))));
                                            let _e10599_ = (_e10598_ & 2139095040u);
                                            if (_e10599_ == 2139095040u) {
                                                phi_22693_ = true;
                                            } else {
                                                phi_22693_ = (_e10599_ == 0u);
                                            }
                                            let _e10603_ = phi_22693_;
                                            let _e10614_ = bitcast<u32>(((_e1006_ * _e10477_) * ((_e1061_ * _e9889_.member.member[0]) + (_e790_ * (_e7022_.member[0] * _e7026_.member[0])))));
                                            let _e10615_ = (_e10614_ & 2139095040u);
                                            if (_e10615_ == 2139095040u) {
                                                phi_22717_ = true;
                                            } else {
                                                phi_22717_ = (_e10615_ == 0u);
                                            }
                                            let _e10619_ = phi_22717_;
                                            let _e10622_ = bitcast<u32>(((_e1024_ * _e10479_) * ((_e1062_ * _e9889_.member.member[1]) + (_e808_ * (_e7022_.member[1] * _e7026_.member[1])))));
                                            let _e10623_ = (_e10622_ & 2139095040u);
                                            if (_e10623_ == 2139095040u) {
                                                phi_22733_ = true;
                                            } else {
                                                phi_22733_ = (_e10623_ == 0u);
                                            }
                                            let _e10627_ = phi_22733_;
                                            let _e10630_ = bitcast<u32>(((_e1042_ * _e10481_) * ((_e1063_ * _e9889_.member.member[2]) + (_e826_ * (_e7022_.member[2] * _e7026_.member[2])))));
                                            let _e10631_ = (_e10630_ & 2139095040u);
                                            if (_e10631_ == 2139095040u) {
                                                phi_22749_ = true;
                                            } else {
                                                phi_22749_ = (_e10631_ == 0u);
                                            }
                                            let _e10635_ = phi_22749_;
                                            let _e10638_ = bitcast<u32>(((_e1060_ * _e10483_) * ((_e1064_ * _e9889_.member.member[3]) + (_e844_ * (_e7022_.member[3] * _e7026_.member[3])))));
                                            let _e10639_ = (_e10638_ & 2139095040u);
                                            if (_e10639_ == 2139095040u) {
                                                phi_22765_ = true;
                                            } else {
                                                phi_22765_ = (_e10639_ == 0u);
                                            }
                                            let _e10643_ = phi_22765_;
                                            let _e10659_ = select(0f, _e790_, (bitcast<f32>((bitcast<u32>(_e790_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10661_ = select(0f, _e808_, (bitcast<f32>((bitcast<u32>(_e808_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10663_ = select(0f, _e826_, (bitcast<f32>((bitcast<u32>(_e826_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10665_ = select(0f, _e844_, (bitcast<f32>((bitcast<u32>(_e844_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e10659_ != _e10659_) {
                                                phi_35194_ = true;
                                            } else {
                                                phi_35194_ = (0f >= _e10659_);
                                            }
                                            let _e10669_ = phi_35194_;
                                            let _e10670_ = select(_e10659_, 0f, _e10669_);
                                            if (_e10661_ != _e10661_) {
                                                phi_35209_ = true;
                                            } else {
                                                phi_35209_ = (0f >= _e10661_);
                                            }
                                            let _e10674_ = phi_35209_;
                                            let _e10675_ = select(_e10661_, 0f, _e10674_);
                                            if (_e10663_ != _e10663_) {
                                                phi_35224_ = true;
                                            } else {
                                                phi_35224_ = (0f >= _e10663_);
                                            }
                                            let _e10679_ = phi_35224_;
                                            let _e10680_ = select(_e10663_, 0f, _e10679_);
                                            if (_e10665_ != _e10665_) {
                                                phi_35239_ = true;
                                            } else {
                                                phi_35239_ = (0f >= _e10665_);
                                            }
                                            let _e10684_ = phi_35239_;
                                            let _e10685_ = select(_e10665_, 0f, _e10684_);
                                            if (_e10670_ != _e10670_) {
                                                phi_35254_ = true;
                                            } else {
                                                phi_35254_ = (1f <= _e10670_);
                                            }
                                            let _e10689_ = phi_35254_;
                                            if (_e10675_ != _e10675_) {
                                                phi_35269_ = true;
                                            } else {
                                                phi_35269_ = (1f <= _e10675_);
                                            }
                                            let _e10694_ = phi_35269_;
                                            if (_e10680_ != _e10680_) {
                                                phi_35284_ = true;
                                            } else {
                                                phi_35284_ = (1f <= _e10680_);
                                            }
                                            let _e10699_ = phi_35284_;
                                            if (_e10685_ != _e10685_) {
                                                phi_35299_ = true;
                                            } else {
                                                phi_35299_ = (1f <= _e10685_);
                                            }
                                            let _e10704_ = phi_35299_;
                                            let _e10719_ = select(0f, _e934_, (bitcast<f32>((bitcast<u32>(_e934_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10721_ = select(0f, _e952_, (bitcast<f32>((bitcast<u32>(_e952_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10723_ = select(0f, _e970_, (bitcast<f32>((bitcast<u32>(_e970_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10725_ = select(0f, _e988_, (bitcast<f32>((bitcast<u32>(_e988_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e10719_ != _e10719_) {
                                                phi_35314_ = true;
                                            } else {
                                                phi_35314_ = (-4f >= _e10719_);
                                            }
                                            let _e10729_ = phi_35314_;
                                            let _e10730_ = select(_e10719_, -4f, _e10729_);
                                            if (_e10721_ != _e10721_) {
                                                phi_35329_ = true;
                                            } else {
                                                phi_35329_ = (-4f >= _e10721_);
                                            }
                                            let _e10734_ = phi_35329_;
                                            let _e10735_ = select(_e10721_, -4f, _e10734_);
                                            if (_e10723_ != _e10723_) {
                                                phi_35344_ = true;
                                            } else {
                                                phi_35344_ = (-4f >= _e10723_);
                                            }
                                            let _e10739_ = phi_35344_;
                                            let _e10740_ = select(_e10723_, -4f, _e10739_);
                                            if (_e10725_ != _e10725_) {
                                                phi_35359_ = true;
                                            } else {
                                                phi_35359_ = (-4f >= _e10725_);
                                            }
                                            let _e10744_ = phi_35359_;
                                            let _e10745_ = select(_e10725_, -4f, _e10744_);
                                            if (_e10730_ != _e10730_) {
                                                phi_35374_ = true;
                                            } else {
                                                phi_35374_ = (4f <= _e10730_);
                                            }
                                            let _e10749_ = phi_35374_;
                                            let _e10750_ = select(_e10730_, 4f, _e10749_);
                                            if (_e10735_ != _e10735_) {
                                                phi_35389_ = true;
                                            } else {
                                                phi_35389_ = (4f <= _e10735_);
                                            }
                                            let _e10754_ = phi_35389_;
                                            let _e10755_ = select(_e10735_, 4f, _e10754_);
                                            if (_e10740_ != _e10740_) {
                                                phi_35404_ = true;
                                            } else {
                                                phi_35404_ = (4f <= _e10740_);
                                            }
                                            let _e10759_ = phi_35404_;
                                            let _e10760_ = select(_e10740_, 4f, _e10759_);
                                            if (_e10745_ != _e10745_) {
                                                phi_35419_ = true;
                                            } else {
                                                phi_35419_ = (4f <= _e10745_);
                                            }
                                            let _e10764_ = phi_35419_;
                                            let _e10765_ = select(_e10745_, 4f, _e10764_);
                                            let _e10779_ = select(0f, _e1006_, (bitcast<f32>((bitcast<u32>(_e1006_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10781_ = select(0f, _e1024_, (bitcast<f32>((bitcast<u32>(_e1024_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10783_ = select(0f, _e1042_, (bitcast<f32>((bitcast<u32>(_e1042_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            let _e10785_ = select(0f, _e1060_, (bitcast<f32>((bitcast<u32>(_e1060_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                            if (_e10779_ != _e10779_) {
                                                phi_35434_ = true;
                                            } else {
                                                phi_35434_ = (-4f >= _e10779_);
                                            }
                                            let _e10789_ = phi_35434_;
                                            let _e10790_ = select(_e10779_, -4f, _e10789_);
                                            if (_e10781_ != _e10781_) {
                                                phi_35449_ = true;
                                            } else {
                                                phi_35449_ = (-4f >= _e10781_);
                                            }
                                            let _e10794_ = phi_35449_;
                                            let _e10795_ = select(_e10781_, -4f, _e10794_);
                                            if (_e10783_ != _e10783_) {
                                                phi_35464_ = true;
                                            } else {
                                                phi_35464_ = (-4f >= _e10783_);
                                            }
                                            let _e10799_ = phi_35464_;
                                            let _e10800_ = select(_e10783_, -4f, _e10799_);
                                            if (_e10785_ != _e10785_) {
                                                phi_35479_ = true;
                                            } else {
                                                phi_35479_ = (-4f >= _e10785_);
                                            }
                                            let _e10804_ = phi_35479_;
                                            let _e10805_ = select(_e10785_, -4f, _e10804_);
                                            if (_e10790_ != _e10790_) {
                                                phi_35494_ = true;
                                            } else {
                                                phi_35494_ = (4f <= _e10790_);
                                            }
                                            let _e10809_ = phi_35494_;
                                            if (_e10795_ != _e10795_) {
                                                phi_35509_ = true;
                                            } else {
                                                phi_35509_ = (4f <= _e10795_);
                                            }
                                            let _e10814_ = phi_35509_;
                                            if (_e10800_ != _e10800_) {
                                                phi_35524_ = true;
                                            } else {
                                                phi_35524_ = (4f <= _e10800_);
                                            }
                                            let _e10819_ = phi_35524_;
                                            if (_e10805_ != _e10805_) {
                                                phi_35539_ = true;
                                            } else {
                                                phi_35539_ = (4f <= _e10805_);
                                            }
                                            let _e10824_ = phi_35539_;
                                            let _e10826_ = (_e10205_ * _e8510_.member.member[0]);
                                            let _e10827_ = (_e10207_ * _e8510_.member.member[1]);
                                            let _e10828_ = (_e10209_ * _e8510_.member.member[2]);
                                            let _e10829_ = (_e10211_ * _e8510_.member.member[3]);
                                            let _e10830_ = (_e10477_ * _e9889_.member.member[0]);
                                            let _e10831_ = (_e10479_ * _e9889_.member.member[1]);
                                            let _e10832_ = (_e10481_ * _e9889_.member.member[2]);
                                            let _e10833_ = (_e10483_ * _e9889_.member.member[3]);
                                            let _e10888_ = (select(_e10790_, 4f, _e10809_) * (1f - select(_e10670_, 1f, _e10689_)));
                                            let _e10889_ = (select(_e10795_, 4f, _e10814_) * (1f - select(_e10675_, 1f, _e10694_)));
                                            let _e10890_ = (select(_e10800_, 4f, _e10819_) * (1f - select(_e10680_, 1f, _e10699_)));
                                            let _e10891_ = (select(_e10805_, 4f, _e10824_) * (1f - select(_e10685_, 1f, _e10704_)));
                                            let _e10900_ = bitcast<u32>((_e10888_ * ((_e10205_ * _e8510_.member_1_.member[0]) + (_e10750_ * (_e10826_ * _e10827_)))));
                                            let _e10901_ = (_e10900_ & 2139095040u);
                                            if (_e10901_ == 2139095040u) {
                                                phi_23222_ = true;
                                            } else {
                                                phi_23222_ = (_e10901_ == 0u);
                                            }
                                            let _e10905_ = phi_23222_;
                                            let _e10908_ = bitcast<u32>((_e10889_ * ((_e10207_ * _e8510_.member_1_.member[1]) + (_e10755_ * (_e10827_ * _e10828_)))));
                                            let _e10909_ = (_e10908_ & 2139095040u);
                                            if (_e10909_ == 2139095040u) {
                                                phi_23239_ = true;
                                            } else {
                                                phi_23239_ = (_e10909_ == 0u);
                                            }
                                            let _e10913_ = phi_23239_;
                                            let _e10916_ = bitcast<u32>((_e10890_ * ((_e10209_ * _e8510_.member_1_.member[2]) + (_e10760_ * (_e10828_ * _e10829_)))));
                                            let _e10917_ = (_e10916_ & 2139095040u);
                                            if (_e10917_ == 2139095040u) {
                                                phi_23256_ = true;
                                            } else {
                                                phi_23256_ = (_e10917_ == 0u);
                                            }
                                            let _e10921_ = phi_23256_;
                                            let _e10924_ = bitcast<u32>((_e10891_ * ((_e10211_ * _e8510_.member_1_.member[3]) + (_e10765_ * (_e10829_ * _e10826_)))));
                                            let _e10925_ = (_e10924_ & 2139095040u);
                                            if (_e10925_ == 2139095040u) {
                                                phi_23273_ = true;
                                            } else {
                                                phi_23273_ = (_e10925_ == 0u);
                                            }
                                            let _e10929_ = phi_23273_;
                                            let _e10932_ = bitcast<u32>((_e10888_ * ((_e10477_ * _e9889_.member_1_.member[0]) + (_e10750_ * (_e10830_ * _e10831_)))));
                                            let _e10933_ = (_e10932_ & 2139095040u);
                                            if (_e10933_ == 2139095040u) {
                                                phi_23293_ = true;
                                            } else {
                                                phi_23293_ = (_e10933_ == 0u);
                                            }
                                            let _e10937_ = phi_23293_;
                                            let _e10940_ = bitcast<u32>((_e10889_ * ((_e10479_ * _e9889_.member_1_.member[1]) + (_e10755_ * (_e10831_ * _e10832_)))));
                                            let _e10941_ = (_e10940_ & 2139095040u);
                                            if (_e10941_ == 2139095040u) {
                                                phi_23310_ = true;
                                            } else {
                                                phi_23310_ = (_e10941_ == 0u);
                                            }
                                            let _e10945_ = phi_23310_;
                                            let _e10948_ = bitcast<u32>((_e10890_ * ((_e10481_ * _e9889_.member_1_.member[2]) + (_e10760_ * (_e10832_ * _e10833_)))));
                                            let _e10949_ = (_e10948_ & 2139095040u);
                                            if (_e10949_ == 2139095040u) {
                                                phi_23327_ = true;
                                            } else {
                                                phi_23327_ = (_e10949_ == 0u);
                                            }
                                            let _e10953_ = phi_23327_;
                                            let _e10956_ = bitcast<u32>((_e10891_ * ((_e10483_ * _e9889_.member_1_.member[3]) + (_e10765_ * (_e10833_ * _e10830_)))));
                                            let _e10957_ = (_e10956_ & 2139095040u);
                                            if (_e10957_ == 2139095040u) {
                                                phi_23344_ = true;
                                            } else {
                                                phi_23344_ = (_e10957_ == 0u);
                                            }
                                            let _e10961_ = phi_23344_;
                                            let _e10964_ = (_e2409_ * 6u);
                                            let _e10969_ = bitcast<u32>(_e7032_.member[0]);
                                            let _e10970_ = (_e10969_ & 2139095040u);
                                            if (_e10970_ == 2139095040u) {
                                                phi_23377_ = true;
                                            } else {
                                                phi_23377_ = (_e10970_ == 0u);
                                            }
                                            let _e10974_ = phi_23377_;
                                            let _e10977_ = bitcast<u32>(_e7032_.member[1]);
                                            let _e10978_ = (_e10977_ & 2139095040u);
                                            if (_e10978_ == 2139095040u) {
                                                phi_23393_ = true;
                                            } else {
                                                phi_23393_ = (_e10978_ == 0u);
                                            }
                                            let _e10982_ = phi_23393_;
                                            let _e10985_ = bitcast<u32>(_e7032_.member[2]);
                                            let _e10986_ = (_e10985_ & 2139095040u);
                                            if (_e10986_ == 2139095040u) {
                                                phi_23409_ = true;
                                            } else {
                                                phi_23409_ = (_e10986_ == 0u);
                                            }
                                            let _e10990_ = phi_23409_;
                                            let _e10993_ = bitcast<u32>(_e7032_.member[3]);
                                            let _e10994_ = (_e10993_ & 2139095040u);
                                            if (_e10994_ == 2139095040u) {
                                                phi_23425_ = true;
                                            } else {
                                                phi_23425_ = (_e10994_ == 0u);
                                            }
                                            let _e10998_ = phi_23425_;
                                            let _e11002_ = (_e10964_ < _e195_);
                                            if _e11002_ {
                                                if _e11002_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e10964_] = vec4<f32>(bitcast<f32>(select(_e10969_, 0u, _e10974_)), bitcast<f32>(select(_e10977_, 0u, _e10982_)), bitcast<f32>(select(_e10985_, 0u, _e10990_)), bitcast<f32>(select(_e10993_, 0u, _e10998_)));
                                            }
                                            let _e11005_ = (_e10964_ + 1u);
                                            let _e11010_ = bitcast<u32>(_e7030_.member[0]);
                                            let _e11011_ = (_e11010_ & 2139095040u);
                                            if (_e11011_ == 2139095040u) {
                                                phi_23458_ = true;
                                            } else {
                                                phi_23458_ = (_e11011_ == 0u);
                                            }
                                            let _e11015_ = phi_23458_;
                                            let _e11018_ = bitcast<u32>(_e7030_.member[1]);
                                            let _e11019_ = (_e11018_ & 2139095040u);
                                            if (_e11019_ == 2139095040u) {
                                                phi_23474_ = true;
                                            } else {
                                                phi_23474_ = (_e11019_ == 0u);
                                            }
                                            let _e11023_ = phi_23474_;
                                            let _e11026_ = bitcast<u32>(_e7030_.member[2]);
                                            let _e11027_ = (_e11026_ & 2139095040u);
                                            if (_e11027_ == 2139095040u) {
                                                phi_23490_ = true;
                                            } else {
                                                phi_23490_ = (_e11027_ == 0u);
                                            }
                                            let _e11031_ = phi_23490_;
                                            let _e11034_ = bitcast<u32>(_e7030_.member[3]);
                                            let _e11035_ = (_e11034_ & 2139095040u);
                                            if (_e11035_ == 2139095040u) {
                                                phi_23506_ = true;
                                            } else {
                                                phi_23506_ = (_e11035_ == 0u);
                                            }
                                            let _e11039_ = phi_23506_;
                                            let _e11043_ = (_e11005_ < _e195_);
                                            if _e11043_ {
                                                if _e11043_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e11005_] = vec4<f32>(bitcast<f32>(select(_e11010_, 0u, _e11015_)), bitcast<f32>(select(_e11018_, 0u, _e11023_)), bitcast<f32>(select(_e11026_, 0u, _e11031_)), bitcast<f32>(select(_e11034_, 0u, _e11039_)));
                                            }
                                            let _e11046_ = (_e10964_ + 2u);
                                            let _e11047_ = bitcast<u32>(bitcast<f32>(select(_e10574_, 0u, _e10579_)));
                                            let _e11048_ = (_e11047_ & 2139095040u);
                                            if (_e11048_ == 2139095040u) {
                                                phi_23534_ = true;
                                            } else {
                                                phi_23534_ = (_e11048_ == 0u);
                                            }
                                            let _e11052_ = phi_23534_;
                                            let _e11055_ = bitcast<u32>(bitcast<f32>(select(_e10582_, 0u, _e10587_)));
                                            let _e11056_ = (_e11055_ & 2139095040u);
                                            if (_e11056_ == 2139095040u) {
                                                phi_23550_ = true;
                                            } else {
                                                phi_23550_ = (_e11056_ == 0u);
                                            }
                                            let _e11060_ = phi_23550_;
                                            let _e11063_ = bitcast<u32>(bitcast<f32>(select(_e10590_, 0u, _e10595_)));
                                            let _e11064_ = (_e11063_ & 2139095040u);
                                            if (_e11064_ == 2139095040u) {
                                                phi_23566_ = true;
                                            } else {
                                                phi_23566_ = (_e11064_ == 0u);
                                            }
                                            let _e11068_ = phi_23566_;
                                            let _e11071_ = bitcast<u32>(bitcast<f32>(select(_e10598_, 0u, _e10603_)));
                                            let _e11072_ = (_e11071_ & 2139095040u);
                                            if (_e11072_ == 2139095040u) {
                                                phi_23582_ = true;
                                            } else {
                                                phi_23582_ = (_e11072_ == 0u);
                                            }
                                            let _e11076_ = phi_23582_;
                                            let _e11080_ = (_e11046_ < _e195_);
                                            if _e11080_ {
                                                if _e11080_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e11046_] = vec4<f32>(bitcast<f32>(select(_e11047_, 0u, _e11052_)), bitcast<f32>(select(_e11055_, 0u, _e11060_)), bitcast<f32>(select(_e11063_, 0u, _e11068_)), bitcast<f32>(select(_e11071_, 0u, _e11076_)));
                                            }
                                            let _e11083_ = (_e10964_ + 3u);
                                            let _e11084_ = bitcast<u32>(bitcast<f32>(select(_e10614_, 0u, _e10619_)));
                                            let _e11085_ = (_e11084_ & 2139095040u);
                                            if (_e11085_ == 2139095040u) {
                                                phi_23610_ = true;
                                            } else {
                                                phi_23610_ = (_e11085_ == 0u);
                                            }
                                            let _e11089_ = phi_23610_;
                                            let _e11092_ = bitcast<u32>(bitcast<f32>(select(_e10622_, 0u, _e10627_)));
                                            let _e11093_ = (_e11092_ & 2139095040u);
                                            if (_e11093_ == 2139095040u) {
                                                phi_23626_ = true;
                                            } else {
                                                phi_23626_ = (_e11093_ == 0u);
                                            }
                                            let _e11097_ = phi_23626_;
                                            let _e11100_ = bitcast<u32>(bitcast<f32>(select(_e10630_, 0u, _e10635_)));
                                            let _e11101_ = (_e11100_ & 2139095040u);
                                            if (_e11101_ == 2139095040u) {
                                                phi_23642_ = true;
                                            } else {
                                                phi_23642_ = (_e11101_ == 0u);
                                            }
                                            let _e11105_ = phi_23642_;
                                            let _e11108_ = bitcast<u32>(bitcast<f32>(select(_e10638_, 0u, _e10643_)));
                                            let _e11109_ = (_e11108_ & 2139095040u);
                                            if (_e11109_ == 2139095040u) {
                                                phi_23658_ = true;
                                            } else {
                                                phi_23658_ = (_e11109_ == 0u);
                                            }
                                            let _e11113_ = phi_23658_;
                                            let _e11117_ = (_e11083_ < _e195_);
                                            if _e11117_ {
                                                if _e11117_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e11083_] = vec4<f32>(bitcast<f32>(select(_e11084_, 0u, _e11089_)), bitcast<f32>(select(_e11092_, 0u, _e11097_)), bitcast<f32>(select(_e11100_, 0u, _e11105_)), bitcast<f32>(select(_e11108_, 0u, _e11113_)));
                                            }
                                            let _e11120_ = (_e10964_ + 4u);
                                            let _e11121_ = bitcast<u32>(bitcast<f32>(select(_e10900_, 0u, _e10905_)));
                                            let _e11122_ = (_e11121_ & 2139095040u);
                                            if (_e11122_ == 2139095040u) {
                                                phi_23690_ = true;
                                            } else {
                                                phi_23690_ = (_e11122_ == 0u);
                                            }
                                            let _e11126_ = phi_23690_;
                                            let _e11129_ = bitcast<u32>(bitcast<f32>(select(_e10908_, 0u, _e10913_)));
                                            let _e11130_ = (_e11129_ & 2139095040u);
                                            if (_e11130_ == 2139095040u) {
                                                phi_23706_ = true;
                                            } else {
                                                phi_23706_ = (_e11130_ == 0u);
                                            }
                                            let _e11134_ = phi_23706_;
                                            let _e11137_ = bitcast<u32>(bitcast<f32>(select(_e10916_, 0u, _e10921_)));
                                            let _e11138_ = (_e11137_ & 2139095040u);
                                            if (_e11138_ == 2139095040u) {
                                                phi_23722_ = true;
                                            } else {
                                                phi_23722_ = (_e11138_ == 0u);
                                            }
                                            let _e11142_ = phi_23722_;
                                            let _e11145_ = bitcast<u32>(bitcast<f32>(select(_e10924_, 0u, _e10929_)));
                                            let _e11146_ = (_e11145_ & 2139095040u);
                                            if (_e11146_ == 2139095040u) {
                                                phi_23738_ = true;
                                            } else {
                                                phi_23738_ = (_e11146_ == 0u);
                                            }
                                            let _e11150_ = phi_23738_;
                                            let _e11154_ = (_e11120_ < _e195_);
                                            if _e11154_ {
                                                if _e11154_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e11120_] = vec4<f32>(bitcast<f32>(select(_e11121_, 0u, _e11126_)), bitcast<f32>(select(_e11129_, 0u, _e11134_)), bitcast<f32>(select(_e11137_, 0u, _e11142_)), bitcast<f32>(select(_e11145_, 0u, _e11150_)));
                                            }
                                            let _e11157_ = (_e10964_ + 5u);
                                            let _e11158_ = bitcast<u32>(bitcast<f32>(select(_e10932_, 0u, _e10937_)));
                                            let _e11159_ = (_e11158_ & 2139095040u);
                                            if (_e11159_ == 2139095040u) {
                                                phi_23770_ = true;
                                            } else {
                                                phi_23770_ = (_e11159_ == 0u);
                                            }
                                            let _e11163_ = phi_23770_;
                                            let _e11166_ = bitcast<u32>(bitcast<f32>(select(_e10940_, 0u, _e10945_)));
                                            let _e11167_ = (_e11166_ & 2139095040u);
                                            if (_e11167_ == 2139095040u) {
                                                phi_23786_ = true;
                                            } else {
                                                phi_23786_ = (_e11167_ == 0u);
                                            }
                                            let _e11171_ = phi_23786_;
                                            let _e11174_ = bitcast<u32>(bitcast<f32>(select(_e10948_, 0u, _e10953_)));
                                            let _e11175_ = (_e11174_ & 2139095040u);
                                            if (_e11175_ == 2139095040u) {
                                                phi_23802_ = true;
                                            } else {
                                                phi_23802_ = (_e11175_ == 0u);
                                            }
                                            let _e11179_ = phi_23802_;
                                            let _e11182_ = bitcast<u32>(bitcast<f32>(select(_e10956_, 0u, _e10961_)));
                                            let _e11183_ = (_e11182_ & 2139095040u);
                                            if (_e11183_ == 2139095040u) {
                                                phi_23818_ = true;
                                            } else {
                                                phi_23818_ = (_e11183_ == 0u);
                                            }
                                            let _e11187_ = phi_23818_;
                                            let _e11191_ = (_e11157_ < _e195_);
                                            if _e11191_ {
                                                if _e11191_ {
                                                } else {
                                                    phi_38285_ = true;
                                                    break;
                                                }
                                                feature_outputs.member[_e11157_] = vec4<f32>(bitcast<f32>(select(_e11158_, 0u, _e11163_)), bitcast<f32>(select(_e11166_, 0u, _e11171_)), bitcast<f32>(select(_e11174_, 0u, _e11179_)), bitcast<f32>(select(_e11182_, 0u, _e11187_)));
                                            }
                                            if ((_e197_ + 1u) == _e2404_) {
                                                local_21_ = array<f32, 4>(_e8510_.member_2_.member[0], _e8510_.member_2_.member[1], _e8510_.member_2_.member[2], _e8510_.member_2_.member[3]);
                                                local_22_ = array<f32, 4>(_e9889_.member_2_.member[0], _e9889_.member_2_.member[1], _e9889_.member_2_.member[2], _e9889_.member_2_.member[3]);
                                                phi_23852_ = type_15_(0u, 4u);
                                                loop {
                                                    let _e11207_ = phi_23852_;
                                                    if (_e11207_.member < _e11207_.member_1_) {
                                                        phi_23853_ = type_15_((_e11207_.member + 1u), _e11207_.member_1_);
                                                        phi_23876_ = type_15_(1u, _e11207_.member);
                                                    } else {
                                                        phi_23853_ = _e11207_;
                                                        phi_23876_ = type_15_(0u, u32());
                                                    }
                                                    let _e11220_ = phi_23853_;
                                                    let _e11222_ = phi_23876_;
                                                    let _e11226_ = (bitcast<i32>(_e11222_.member) != 0i);
                                                    if _e11226_ {
                                                        let _e11227_ = (_e11222_.member_1_ < 4u);
                                                        if _e11227_ {
                                                        } else {
                                                            phi_37314_ = true;
                                                            break;
                                                        }
                                                        let _e11229_ = local_21_[_e11222_.member_1_];
                                                        let _e11230_ = (24u + _e11222_.member_1_);
                                                        if (_e11230_ < 256u) {
                                                        } else {
                                                            phi_37314_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e11230_] = _e11229_;
                                                        if _e11227_ {
                                                        } else {
                                                            phi_37314_ = true;
                                                            break;
                                                        }
                                                        let _e11234_ = local_22_[_e11222_.member_1_];
                                                        let _e11235_ = (28u + _e11222_.member_1_);
                                                        if (_e11235_ < 256u) {
                                                        } else {
                                                            phi_37314_ = true;
                                                            break;
                                                        }
                                                        aux_scratch[_e11235_] = _e11234_;
                                                    }
                                                    phi_23852_ = _e11220_;
                                                    phi_37314_ = _e7109_;
                                                    if !(_e11226_) {
                                                        break;
                                                    }
                                                }
                                                let _e11240_ = phi_37314_;
                                                phi_38285_ = _e11240_;
                                                if _e11240_ {
                                                    break;
                                                }
                                                phi_38290_ = _e11240_;
                                            } else {
                                                phi_38290_ = _e7109_;
                                            }
                                            let _e11242_ = phi_38290_;
                                            phi_38289_ = _e11242_;
                                        } else {
                                            phi_38289_ = _e7016_;
                                        }
                                        let _e11244_ = phi_38289_;
                                        phi_38288_ = _e11244_;
                                    } else {
                                        phi_38288_ = _e7016_;
                                    }
                                    let _e11246_ = phi_38288_;
                                    workgroupBarrier();
                                    phi_38287_ = _e11246_;
                                } else {
                                    phi_38287_ = _e2375_;
                                }
                                let _e11248_ = phi_38287_;
                                phi_38286_ = _e11248_;
                            } else {
                                phi_38286_ = _e2375_;
                            }
                            let _e11250_ = phi_38286_;
                            phi_36006_ = _e11250_;
                            phi_5466_ = _e2390_;
                            phi_38285_ = _e11250_;
                            if !(_e2396_) {
                                break;
                            }
                        }
                        let _e11253_ = phi_38285_;
                        phi_38291_ = _e11253_;
                        if _e11253_ {
                            break;
                        }
                        phi_38292_ = _e11253_;
                    } else {
                        phi_38292_ = _e1804_;
                    }
                    let _e11255_ = phi_38292_;
                    phi_36009_ = _e11255_;
                    phi_4129_ = _e1819_;
                    phi_38291_ = _e11255_;
                    if !(_e1825_) {
                        break;
                    }
                }
                let _e11258_ = phi_38291_;
                if _e11258_ {
                    break;
                }
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
