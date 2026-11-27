struct type_6_ {
    member: array<f32>,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
}

struct type_20_ {
    member: f32,
    member_1_: f32,
}

struct type_22_ {
    member: array<f32, 4>,
}

struct type_23_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_25_ {
    member: array<vec2<f32>>,
}

struct type_27_ {
    member: type_22_,
    member_1_: type_22_,
    member_2_: type_22_,
}

struct type_28_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: u32,
    member_7_: u32,
}

struct type_29_ {
    member: type_28_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> sg_lid_1_: u32;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(2) 
var<storage> target_audio: type_25_;
@group(0) @binding(0) 
var<uniform> uniforms: type_29_;
@group(0) @binding(1) 
var<storage> input_audio: type_25_;
@group(0) @binding(3) 
var<storage> seed_params: type_6_;
@group(0) @binding(4) 
var<storage, read_write> seed_losses: type_6_;
var<workgroup> err_scratch: array<f32, 256>;
var<workgroup> tgt_scratch: array<f32, 256>;

fn function_() {
    var phi_224_: u32;
    var phi_231_: u32;
    var phi_254_: u32;
    var phi_277_: bool;
    var phi_293_: bool;
    var phi_301_: vec2<f32>;
    var phi_12014_: bool;
    var phi_304_: type_15_;
    var phi_307_: f32;
    var phi_309_: f32;
    var phi_305_: type_15_;
    var phi_332_: type_15_;
    var phi_350_: u32;
    var phi_361_: u32;
    var phi_373_: u32;
    var phi_392_: u32;
    var phi_425_: f32;
    var phi_431_: f32;
    var phi_8138_: bool;
    var phi_602_: bool;
    var phi_608_: f32;
    var phi_609_: f32;
    var phi_528_: f32;
    var phi_530_: f32;
    var phi_531_: bool;
    var phi_551_: f32;
    var phi_610_: f32;
    var phi_611_: f32;
    var phi_8198_: bool;
    var phi_697_: bool;
    var phi_8258_: bool;
    var phi_662_: bool;
    var phi_703_: f32;
    var phi_704_: f32;
    var phi_705_: f32;
    var phi_706_: bool;
    var phi_722_: u32;
    var phi_740_: bool;
    var phi_757_: u32;
    var phi_775_: bool;
    var phi_792_: u32;
    var phi_810_: bool;
    var phi_827_: u32;
    var phi_845_: bool;
    var phi_862_: u32;
    var phi_880_: bool;
    var phi_897_: u32;
    var phi_915_: bool;
    var phi_8318_: bool;
    var phi_8378_: bool;
    var phi_1496_: bool;
    var phi_1502_: f32;
    var phi_1503_: f32;
    var phi_1422_: f32;
    var phi_1424_: f32;
    var phi_1425_: bool;
    var phi_1504_: f32;
    var phi_1516_: u32;
    var phi_1534_: bool;
    var phi_1551_: u32;
    var phi_1569_: bool;
    var phi_1586_: u32;
    var phi_1604_: bool;
    var phi_1621_: u32;
    var phi_1639_: bool;
    var phi_1656_: u32;
    var phi_1780_: bool;
    var phi_1786_: f32;
    var phi_1787_: f32;
    var phi_1706_: f32;
    var phi_1708_: f32;
    var phi_1709_: bool;
    var phi_1729_: f32;
    var phi_1788_: f32;
    var phi_1800_: u32;
    var phi_1924_: bool;
    var phi_1930_: f32;
    var phi_1931_: f32;
    var phi_1850_: f32;
    var phi_1852_: f32;
    var phi_1853_: bool;
    var phi_1873_: f32;
    var phi_1932_: f32;
    var phi_1944_: u32;
    var phi_2068_: bool;
    var phi_2074_: f32;
    var phi_2075_: f32;
    var phi_1994_: f32;
    var phi_1996_: f32;
    var phi_1997_: bool;
    var phi_2017_: f32;
    var phi_2076_: f32;
    var phi_2088_: u32;
    var phi_2212_: bool;
    var phi_2218_: f32;
    var phi_2219_: f32;
    var phi_2138_: f32;
    var phi_2140_: f32;
    var phi_2141_: bool;
    var phi_2161_: f32;
    var phi_2220_: f32;
    var phi_2242_: type_23_;
    var phi_2302_: u32;
    var phi_2303_: u32;
    var phi_11818_: bool;
    var phi_2565_: u32;
    var phi_2566_: u32;
    var phi_12119_: bool;
    var phi_3108_: bool;
    var phi_3124_: bool;
    var phi_3132_: type_20_;
    var phi_3155_: type_23_;
    var phi_3215_: u32;
    var phi_3216_: u32;
    var phi_12134_: bool;
    var phi_3478_: u32;
    var phi_3479_: u32;
    var phi_12154_: bool;
    var phi_4021_: bool;
    var phi_4037_: bool;
    var phi_4045_: type_20_;
    var phi_4110_: type_15_;
    var phi_4113_: type_22_;
    var phi_4115_: type_22_;
    var phi_4111_: type_15_;
    var phi_4138_: type_15_;
    var phi_4191_: bool;
    var phi_4207_: bool;
    var phi_4223_: bool;
    var phi_4239_: bool;
    var phi_8438_: bool;
    var phi_8453_: bool;
    var phi_8468_: bool;
    var phi_8483_: bool;
    var phi_8498_: bool;
    var phi_8513_: bool;
    var phi_8528_: bool;
    var phi_8543_: bool;
    var phi_8558_: bool;
    var phi_8573_: bool;
    var phi_8588_: bool;
    var phi_8603_: bool;
    var phi_8618_: bool;
    var phi_8633_: bool;
    var phi_8648_: bool;
    var phi_8663_: bool;
    var phi_8678_: bool;
    var phi_8693_: bool;
    var phi_8708_: bool;
    var phi_8723_: bool;
    var phi_8738_: bool;
    var phi_8753_: bool;
    var phi_8768_: bool;
    var phi_8783_: bool;
    var phi_8798_: bool;
    var phi_8813_: bool;
    var phi_8828_: bool;
    var phi_8843_: bool;
    var phi_8858_: bool;
    var phi_8873_: bool;
    var phi_8888_: bool;
    var phi_8903_: bool;
    var phi_4947_: bool;
    var phi_4963_: bool;
    var phi_4979_: bool;
    var phi_4995_: bool;
    var phi_5017_: bool;
    var phi_5033_: bool;
    var phi_5049_: bool;
    var phi_5065_: bool;
    var phi_9038_: bool;
    var phi_9053_: bool;
    var phi_9068_: bool;
    var phi_9083_: bool;
    var phi_9098_: bool;
    var phi_9113_: bool;
    var phi_9128_: bool;
    var phi_9143_: bool;
    var phi_9158_: bool;
    var phi_9173_: bool;
    var phi_9188_: bool;
    var phi_9203_: bool;
    var phi_9218_: bool;
    var phi_9233_: bool;
    var phi_9248_: bool;
    var phi_9263_: bool;
    var phi_9278_: bool;
    var phi_9293_: bool;
    var phi_9308_: bool;
    var phi_9323_: bool;
    var phi_9338_: bool;
    var phi_9353_: bool;
    var phi_9368_: bool;
    var phi_9383_: bool;
    var phi_9398_: bool;
    var phi_9413_: bool;
    var phi_9428_: bool;
    var phi_9443_: bool;
    var phi_9458_: bool;
    var phi_9473_: bool;
    var phi_9488_: bool;
    var phi_9503_: bool;
    var phi_5773_: bool;
    var phi_5789_: bool;
    var phi_5805_: bool;
    var phi_5821_: bool;
    var phi_4114_: type_22_;
    var phi_4116_: type_22_;
    var phi_12169_: bool;
    var phi_12851_: bool;
    var phi_5909_: f32;
    var phi_5921_: f32;
    var phi_9638_: bool;
    var phi_9653_: bool;
    var phi_5945_: bool;
    var phi_5951_: f32;
    var phi_5969_: bool;
    var phi_5975_: f32;
    var phi_6004_: f32;
    var phi_6006_: f32;
    var phi_6007_: bool;
    var phi_6028_: f32;
    var phi_6030_: f32;
    var phi_6031_: bool;
    var phi_9668_: bool;
    var phi_9683_: bool;
    var phi_9698_: bool;
    var phi_9713_: bool;
    var phi_9728_: bool;
    var phi_9743_: bool;
    var phi_9758_: bool;
    var phi_9773_: bool;
    var phi_9788_: bool;
    var phi_9803_: bool;
    var phi_9818_: bool;
    var phi_9833_: bool;
    var phi_9848_: bool;
    var phi_9863_: bool;
    var phi_9878_: bool;
    var phi_9893_: bool;
    var phi_9908_: bool;
    var phi_9923_: bool;
    var phi_9938_: bool;
    var phi_9953_: bool;
    var phi_9968_: bool;
    var phi_9983_: bool;
    var phi_9998_: bool;
    var phi_10013_: bool;
    var phi_10028_: bool;
    var phi_10043_: bool;
    var phi_10058_: bool;
    var phi_10073_: bool;
    var phi_10088_: bool;
    var phi_10103_: bool;
    var phi_10118_: bool;
    var phi_10133_: bool;
    var phi_10148_: bool;
    var phi_10163_: bool;
    var phi_10178_: bool;
    var phi_10193_: bool;
    var phi_10208_: bool;
    var phi_10223_: bool;
    var phi_10238_: bool;
    var phi_10253_: bool;
    var phi_10268_: bool;
    var phi_10283_: bool;
    var phi_10298_: bool;
    var phi_10313_: bool;
    var phi_10328_: bool;
    var phi_10343_: bool;
    var phi_10358_: bool;
    var phi_10373_: bool;
    var phi_10388_: bool;
    var phi_10403_: bool;
    var phi_10418_: bool;
    var phi_10433_: bool;
    var phi_10448_: bool;
    var phi_10463_: bool;
    var phi_10478_: bool;
    var phi_10493_: bool;
    var phi_10508_: bool;
    var phi_10523_: bool;
    var phi_10538_: bool;
    var phi_10553_: bool;
    var phi_10568_: bool;
    var phi_10583_: bool;
    var phi_10598_: bool;
    var phi_10613_: bool;
    var phi_6458_: bool;
    var phi_6460_: bool;
    var phi_6461_: bool;
    var phi_6462_: bool;
    var phi_6463_: bool;
    var phi_10628_: bool;
    var phi_10643_: bool;
    var phi_10658_: bool;
    var phi_10673_: bool;
    var phi_6841_: type_27_;
    var phi_10688_: bool;
    var phi_10703_: bool;
    var phi_10718_: bool;
    var phi_10733_: bool;
    var phi_10748_: bool;
    var phi_10763_: bool;
    var phi_10778_: bool;
    var phi_10793_: bool;
    var phi_10808_: bool;
    var phi_10823_: bool;
    var phi_10838_: bool;
    var phi_10853_: bool;
    var phi_10868_: bool;
    var phi_10883_: bool;
    var phi_10898_: bool;
    var phi_10913_: bool;
    var phi_10928_: bool;
    var phi_10943_: bool;
    var phi_10958_: bool;
    var phi_10973_: bool;
    var phi_10988_: bool;
    var phi_11003_: bool;
    var phi_11018_: bool;
    var phi_11033_: bool;
    var phi_11048_: bool;
    var phi_11063_: bool;
    var phi_11078_: bool;
    var phi_11093_: bool;
    var phi_11108_: bool;
    var phi_11123_: bool;
    var phi_11138_: bool;
    var phi_11153_: bool;
    var phi_11168_: bool;
    var phi_11183_: bool;
    var phi_11198_: bool;
    var phi_11213_: bool;
    var phi_11228_: bool;
    var phi_11243_: bool;
    var phi_11258_: bool;
    var phi_11273_: bool;
    var phi_11288_: bool;
    var phi_11303_: bool;
    var phi_11318_: bool;
    var phi_11333_: bool;
    var phi_11348_: bool;
    var phi_11363_: bool;
    var phi_11378_: bool;
    var phi_11393_: bool;
    var phi_11408_: bool;
    var phi_11423_: bool;
    var phi_11438_: bool;
    var phi_11453_: bool;
    var phi_11468_: bool;
    var phi_11483_: bool;
    var phi_11498_: bool;
    var phi_11513_: bool;
    var phi_11528_: bool;
    var phi_11543_: bool;
    var phi_11558_: bool;
    var phi_11573_: bool;
    var phi_11588_: bool;
    var phi_11603_: bool;
    var phi_11618_: bool;
    var phi_11633_: bool;
    var phi_7265_: bool;
    var phi_7267_: bool;
    var phi_7268_: bool;
    var phi_7269_: bool;
    var phi_7270_: bool;
    var phi_11648_: bool;
    var phi_11663_: bool;
    var phi_11678_: bool;
    var phi_11693_: bool;
    var phi_7648_: type_27_;
    var phi_7657_: f32;
    var phi_7658_: f32;
    var phi_12455_: bool;
    var phi_308_: f32;
    var phi_310_: f32;
    var phi_12454_: bool;
    var phi_7672_: u32;
    var phi_7695_: bool;
    var phi_7711_: bool;
    var phi_7719_: vec2<f32>;
    var local: f32;
    var phi_7725_: f32;
    var local_1_: f32;
    var phi_7731_: f32;
    var phi_7748_: bool;
    var phi_7754_: f32;
    var phi_7775_: bool;
    var phi_7781_: f32;
    var phi_7792_: bool;
    var phi_7808_: bool;
    var phi_7846_: u32;
    var phi_7847_: u32;
    var phi_12862_: bool;
    var phi_7907_: bool;
    var phi_7925_: bool;
    var phi_7965_: f32;
    var phi_7966_: bool;
    var phi_11708_: bool;
    var phi_8057_: bool;
    var phi_11768_: bool;
    var phi_8022_: bool;
    var phi_8063_: f32;
    var phi_8064_: f32;
    var phi_8065_: f32;
    var phi_8066_: bool;
    var phi_8082_: bool;
    var phi_8102_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e113_ = wg_id_1_;
            let _e114_ = lid_1_;
            let _e115_ = sg_lid_1_;
            let _e117_ = arrayLength((&input_audio.member));
            let _e119_ = arrayLength((&target_audio.member));
            let _e121_ = arrayLength((&seed_params.member));
            let _e123_ = arrayLength((&seed_losses.member));
            let _e129_ = uniforms.member.member;
            let _e131_ = select(1u, _e129_, (1u < _e129_));
            let _e134_ = uniforms.member.member_1_;
            let _e137_ = uniforms.member.member_2_;
            let _e139_ = select(1u, _e137_, (1u < _e137_));
            let _e142_ = uniforms.member.member_3_;
            if (_e134_ == 0u) {
                phi_224_ = 1u;
            } else {
                phi_224_ = (((_e134_ - 1u) >> bitcast<u32>(8i)) + 1u);
            }
            let _e151_ = phi_224_;
            let _e152_ = (_e113_.y * 256u);
            if (_e134_ > _e152_) {
                phi_231_ = (_e134_ - _e152_);
            } else {
                phi_231_ = 0u;
            }
            let _e156_ = phi_231_;
            let _e159_ = (_e152_ + _e114_);
            let _e160_ = (_e159_ < _e134_);
            let _e162_ = ((_e113_.z * _e134_) + _e159_);
            if _e160_ {
                if (_e117_ == 0u) {
                    phi_254_ = 0u;
                } else {
                    let _e164_ = (_e117_ - 1u);
                    phi_254_ = select(_e164_, _e162_, (_e162_ < _e164_));
                }
                let _e168_ = phi_254_;
                if (_e168_ < _e117_) {
                } else {
                    break;
                }
                let _e173_ = input_audio.member[_e168_].x;
                let _e177_ = input_audio.member[_e168_].y;
                let _e178_ = bitcast<u32>(_e173_);
                let _e179_ = (_e178_ & 2139095040u);
                if (_e179_ == 2139095040u) {
                    phi_277_ = true;
                } else {
                    phi_277_ = (_e179_ == 0u);
                }
                let _e183_ = phi_277_;
                let _e186_ = bitcast<u32>(_e177_);
                let _e187_ = (_e186_ & 2139095040u);
                if (_e187_ == 2139095040u) {
                    phi_293_ = true;
                } else {
                    phi_293_ = (_e187_ == 0u);
                }
                let _e191_ = phi_293_;
                phi_301_ = vec2<f32>(bitcast<f32>(select(_e178_, 0u, _e183_)), bitcast<f32>(select(_e186_, 0u, _e191_)));
            } else {
                phi_301_ = vec2<f32>(0f, 0f);
            }
            let _e196_ = phi_301_;
            phi_12014_ = false;
            phi_304_ = type_15_(0u, _e139_);
            phi_307_ = 0f;
            phi_309_ = 0f;
            loop {
                let _e199_ = phi_12014_;
                let _e201_ = phi_304_;
                let _e203_ = phi_307_;
                let _e205_ = phi_309_;
                local = _e205_;
                local_1_ = _e203_;
                if (_e201_.member < _e201_.member_1_) {
                    phi_305_ = type_15_((_e201_.member + 1u), _e201_.member_1_);
                    phi_332_ = type_15_(1u, _e201_.member);
                } else {
                    phi_305_ = _e201_;
                    phi_332_ = type_15_(0u, u32());
                }
                let _e218_ = phi_305_;
                let _e220_ = phi_332_;
                let _e224_ = (bitcast<i32>(_e220_.member) != 0i);
                if _e224_ {
                    if (_e131_ == 0u) {
                        phi_350_ = 0u;
                    } else {
                        let _e226_ = (_e131_ - 1u);
                        phi_350_ = select(_e226_, _e113_.x, (_e113_.x < _e226_));
                    }
                    let _e230_ = phi_350_;
                    if (_e121_ >= ((_e131_ * _e139_) * 16u)) {
                        phi_361_ = (((_e230_ * _e139_) + _e220_.member_1_) * 16u);
                    } else {
                        phi_361_ = (_e230_ * 16u);
                    }
                    let _e239_ = phi_361_;
                    let _e240_ = (_e121_ == 0u);
                    if _e240_ {
                        phi_373_ = 0u;
                    } else {
                        let _e241_ = (_e121_ - 1u);
                        phi_373_ = select(_e241_, _e239_, (_e239_ < _e241_));
                    }
                    let _e245_ = phi_373_;
                    if (_e245_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e249_ = seed_params.member[_e245_];
                    let _e250_ = (_e239_ + 1u);
                    if _e240_ {
                        phi_392_ = 0u;
                    } else {
                        let _e251_ = (_e121_ - 1u);
                        phi_392_ = select(_e251_, _e250_, (_e250_ < _e251_));
                    }
                    let _e255_ = phi_392_;
                    if (_e255_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e259_ = seed_params.member[_e255_];
                    let _e263_ = select(0.001f, _e249_, ((bitcast<u32>(_e249_) & 2139095040u) != 2139095040u));
                    if (_e263_ != _e263_) {
                        phi_425_ = 0.00001f;
                    } else {
                        let _e266_ = select(_e263_, 0.00001f, (_e263_ < 0.00001f));
                        phi_425_ = select(_e266_, 0.495f, (_e266_ > 0.495f));
                    }
                    let _e270_ = phi_425_;
                    let _e271_ = (_e270_ <= 0.25f);
                    if _e271_ {
                        phi_431_ = _e270_;
                    } else {
                        phi_431_ = (0.5f - _e270_);
                    }
                    let _e274_ = phi_431_;
                    let _e275_ = (3.1415927f * _e274_);
                    let _e276_ = (_e275_ * _e275_);
                    let _e287_ = (_e275_ * (1f + (_e276_ * (-0.16666667f + (_e276_ * (0.008333334f + (_e276_ * (-0.0001984127f + (_e276_ * (0.0000027557319f + (_e276_ * -0.000000025052108f)))))))))));
                    let _e297_ = (1f + (_e276_ * (-0.5f + (_e276_ * (0.041666668f + (_e276_ * (-0.0013888889f + (_e276_ * (0.000024801588f + (_e276_ * -0.0000002755732f))))))))));
                    let _e299_ = select(_e287_, _e297_, _e271_);
                    if (_e299_ != _e299_) {
                        phi_8138_ = true;
                    } else {
                        phi_8138_ = (0.000001f >= _e299_);
                    }
                    let _e303_ = phi_8138_;
                    let _e304_ = select(_e299_, 0.000001f, _e303_);
                    let _e307_ = bitcast<f32>((2129859011u - bitcast<u32>(_e304_)));
                    let _e311_ = (_e307_ + (_e307_ * (1f - (_e304_ * _e307_))));
                    let _e315_ = (_e311_ + (_e311_ * (1f - (_e304_ * _e311_))));
                    let _e320_ = (select(_e297_, _e287_, _e271_) * (_e315_ + (_e315_ * (1f - (_e304_ * _e315_)))));
                    let _e321_ = bitcast<u32>(_e259_);
                    let _e322_ = (_e321_ & 2139095040u);
                    if (_e322_ != 2139095040u) {
                        let _e324_ = (_e322_ == 2139095040u);
                        if _e324_ {
                            if ((_e321_ & 2147483647u) > 2139095040u) {
                                phi_530_ = f32();
                                phi_531_ = true;
                            } else {
                                let _e339_ = ((_e321_ & 2147483648u) == 0u);
                                if _e339_ {
                                    phi_528_ = 64f;
                                } else {
                                    phi_528_ = f32();
                                }
                                let _e341_ = phi_528_;
                                phi_530_ = _e341_;
                                phi_531_ = select(true, false, _e339_);
                            }
                            let _e344_ = phi_530_;
                            let _e346_ = phi_531_;
                            if _e346_ {
                                phi_551_ = 0.05f;
                            } else {
                                phi_551_ = _e344_;
                            }
                            let _e348_ = phi_551_;
                            phi_610_ = _e348_;
                        } else {
                            if (_e259_ < 0.05f) {
                                phi_609_ = 0.05f;
                            } else {
                                if (_e259_ > 64f) {
                                    phi_608_ = 64f;
                                } else {
                                    if _e324_ {
                                        phi_602_ = true;
                                    } else {
                                        phi_602_ = (_e322_ == 0u);
                                    }
                                    let _e329_ = phi_602_;
                                    phi_608_ = bitcast<f32>(select(_e321_, 0u, _e329_));
                                }
                                let _e333_ = phi_608_;
                                phi_609_ = _e333_;
                            }
                            let _e335_ = phi_609_;
                            phi_610_ = _e335_;
                        }
                        let _e350_ = phi_610_;
                        phi_611_ = _e350_;
                    } else {
                        phi_611_ = 0.70710677f;
                    }
                    let _e352_ = phi_611_;
                    if ((bitcast<u32>(_e352_) & 2139095040u) != 2139095040u) {
                        let _e356_ = (_e352_ <= 0f);
                        if _e356_ {
                            phi_704_ = f32();
                        } else {
                            if (_e352_ < 0.00001f) {
                                let _e388_ = (select(0.000000000000000000000001f, _e352_, (_e352_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e388_ != _e388_) {
                                    phi_8258_ = true;
                                } else {
                                    phi_8258_ = (0.000001f >= _e388_);
                                }
                                let _e392_ = phi_8258_;
                                let _e393_ = select(_e388_, 0.000001f, _e392_);
                                let _e396_ = bitcast<f32>((2129859011u - bitcast<u32>(_e393_)));
                                let _e400_ = (_e396_ + (_e396_ * (1f - (_e393_ * _e396_))));
                                let _e404_ = (_e400_ + (_e400_ * (1f - (_e393_ * _e400_))));
                                let _e410_ = bitcast<u32>(((_e404_ + (_e404_ * (1f - (_e393_ * _e404_)))) * 18446744000000000000f));
                                let _e411_ = (_e410_ & 2139095040u);
                                if (_e411_ == 2139095040u) {
                                    phi_662_ = true;
                                } else {
                                    phi_662_ = (_e411_ == 0u);
                                }
                                let _e415_ = phi_662_;
                                phi_703_ = bitcast<f32>(select(_e410_, 0u, _e415_));
                            } else {
                                if (_e352_ != _e352_) {
                                    phi_8198_ = true;
                                } else {
                                    phi_8198_ = (0.000001f >= _e352_);
                                }
                                let _e361_ = phi_8198_;
                                let _e362_ = select(_e352_, 0.000001f, _e361_);
                                let _e365_ = bitcast<f32>((2129859011u - bitcast<u32>(_e362_)));
                                let _e369_ = (_e365_ + (_e365_ * (1f - (_e362_ * _e365_))));
                                let _e373_ = (_e369_ + (_e369_ * (1f - (_e362_ * _e369_))));
                                let _e378_ = bitcast<u32>((_e373_ + (_e373_ * (1f - (_e362_ * _e373_)))));
                                let _e379_ = (_e378_ & 2139095040u);
                                if (_e379_ == 2139095040u) {
                                    phi_697_ = true;
                                } else {
                                    phi_697_ = (_e379_ == 0u);
                                }
                                let _e383_ = phi_697_;
                                phi_703_ = bitcast<f32>(select(_e378_, 0u, _e383_));
                            }
                            let _e419_ = phi_703_;
                            phi_704_ = _e419_;
                        }
                        let _e421_ = phi_704_;
                        phi_705_ = _e421_;
                        phi_706_ = _e356_;
                    } else {
                        phi_705_ = f32();
                        phi_706_ = true;
                    }
                    let _e423_ = phi_705_;
                    let _e425_ = phi_706_;
                    let _e426_ = select(_e423_, 0f, _e425_);
                    let _e427_ = (_e239_ + 6u);
                    if _e240_ {
                        phi_722_ = 0u;
                    } else {
                        let _e428_ = (_e121_ - 1u);
                        phi_722_ = select(_e428_, _e427_, (_e427_ < _e428_));
                    }
                    let _e432_ = phi_722_;
                    if (_e432_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e436_ = seed_params.member[_e432_];
                    let _e437_ = bitcast<u32>(_e436_);
                    let _e438_ = (_e437_ & 2139095040u);
                    if (_e438_ == 2139095040u) {
                        phi_740_ = true;
                    } else {
                        phi_740_ = (_e438_ == 0u);
                    }
                    let _e442_ = phi_740_;
                    let _e444_ = bitcast<f32>(select(_e437_, 0u, _e442_));
                    let _e445_ = (_e239_ + 7u);
                    if _e240_ {
                        phi_757_ = 0u;
                    } else {
                        let _e446_ = (_e121_ - 1u);
                        phi_757_ = select(_e446_, _e445_, (_e445_ < _e446_));
                    }
                    let _e450_ = phi_757_;
                    if (_e450_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e454_ = seed_params.member[_e450_];
                    let _e455_ = bitcast<u32>(_e454_);
                    let _e456_ = (_e455_ & 2139095040u);
                    if (_e456_ == 2139095040u) {
                        phi_775_ = true;
                    } else {
                        phi_775_ = (_e456_ == 0u);
                    }
                    let _e460_ = phi_775_;
                    let _e462_ = bitcast<f32>(select(_e455_, 0u, _e460_));
                    let _e463_ = (_e239_ + 8u);
                    if _e240_ {
                        phi_792_ = 0u;
                    } else {
                        let _e464_ = (_e121_ - 1u);
                        phi_792_ = select(_e464_, _e463_, (_e463_ < _e464_));
                    }
                    let _e468_ = phi_792_;
                    if (_e468_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e472_ = seed_params.member[_e468_];
                    let _e473_ = bitcast<u32>(_e472_);
                    let _e474_ = (_e473_ & 2139095040u);
                    if (_e474_ == 2139095040u) {
                        phi_810_ = true;
                    } else {
                        phi_810_ = (_e474_ == 0u);
                    }
                    let _e478_ = phi_810_;
                    let _e480_ = bitcast<f32>(select(_e473_, 0u, _e478_));
                    let _e481_ = (_e239_ + 9u);
                    if _e240_ {
                        phi_827_ = 0u;
                    } else {
                        let _e482_ = (_e121_ - 1u);
                        phi_827_ = select(_e482_, _e481_, (_e481_ < _e482_));
                    }
                    let _e486_ = phi_827_;
                    if (_e486_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e490_ = seed_params.member[_e486_];
                    let _e491_ = bitcast<u32>(_e490_);
                    let _e492_ = (_e491_ & 2139095040u);
                    if (_e492_ == 2139095040u) {
                        phi_845_ = true;
                    } else {
                        phi_845_ = (_e492_ == 0u);
                    }
                    let _e496_ = phi_845_;
                    let _e498_ = bitcast<f32>(select(_e491_, 0u, _e496_));
                    let _e499_ = (_e239_ + 10u);
                    if _e240_ {
                        phi_862_ = 0u;
                    } else {
                        let _e500_ = (_e121_ - 1u);
                        phi_862_ = select(_e500_, _e499_, (_e499_ < _e500_));
                    }
                    let _e504_ = phi_862_;
                    if (_e504_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e508_ = seed_params.member[_e504_];
                    let _e509_ = bitcast<u32>(_e508_);
                    let _e510_ = (_e509_ & 2139095040u);
                    if (_e510_ == 2139095040u) {
                        phi_880_ = true;
                    } else {
                        phi_880_ = (_e510_ == 0u);
                    }
                    let _e514_ = phi_880_;
                    let _e516_ = bitcast<f32>(select(_e509_, 0u, _e514_));
                    let _e517_ = (_e239_ + 11u);
                    if _e240_ {
                        phi_897_ = 0u;
                    } else {
                        let _e518_ = (_e121_ - 1u);
                        phi_897_ = select(_e518_, _e517_, (_e517_ < _e518_));
                    }
                    let _e522_ = phi_897_;
                    if (_e522_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e526_ = seed_params.member[_e522_];
                    let _e527_ = bitcast<u32>(_e526_);
                    let _e528_ = (_e527_ & 2139095040u);
                    if (_e528_ == 2139095040u) {
                        phi_915_ = true;
                    } else {
                        phi_915_ = (_e528_ == 0u);
                    }
                    let _e532_ = phi_915_;
                    let _e534_ = bitcast<f32>(select(_e527_, 0u, _e532_));
                    let _e540_ = select(select(select(0f, 0.999995f, (0.999995f <= _e444_)), _e444_, (_e444_ <= 0.999995f)), 0f, (_e444_ <= 0f));
                    let _e558_ = select(select(select(0f, 64f, (64f <= _e498_)), _e498_, (_e498_ <= 64f)), 0f, (_e498_ <= 0f));
                    let _e564_ = select(select(select(0f, 1f, (1f <= _e516_)), _e516_, (_e516_ <= 1f)), 0f, (_e516_ <= 0f));
                    let _e570_ = select(select(select(0f, 1f, (1f <= _e534_)), _e534_, (_e534_ <= 1f)), 0f, (_e534_ <= 0f));
                    if (_e320_ != _e320_) {
                        phi_8318_ = true;
                    } else {
                        phi_8318_ = (0.00001f >= _e320_);
                    }
                    let _e574_ = phi_8318_;
                    let _e575_ = select(_e320_, 0.00001f, _e574_);
                    if (_e426_ != _e426_) {
                        phi_8378_ = true;
                    } else {
                        phi_8378_ = (0f >= _e426_);
                    }
                    let _e579_ = phi_8378_;
                    let _e580_ = select(_e426_, 0f, _e579_);
                    let _e583_ = (1f + (_e575_ * (_e575_ + _e580_)));
                    let _e586_ = bitcast<f32>((2129859011u - bitcast<u32>(_e583_)));
                    let _e590_ = (_e586_ + (_e586_ * (1f - (_e583_ * _e586_))));
                    let _e594_ = (_e590_ + (_e590_ * (1f - (_e583_ * _e590_))));
                    let _e595_ = (_e575_ * _e594_);
                    let _e596_ = (_e575_ * _e595_);
                    let _e597_ = (_e320_ + _e320_);
                    let _e598_ = bitcast<u32>(_e597_);
                    let _e599_ = (_e598_ & 2139095040u);
                    let _e600_ = (_e599_ == 2139095040u);
                    if _e600_ {
                        if ((_e598_ & 2147483647u) > 2139095040u) {
                            phi_1424_ = f32();
                            phi_1425_ = true;
                        } else {
                            let _e615_ = ((_e598_ & 2147483648u) == 0u);
                            if _e615_ {
                                phi_1422_ = 128f;
                            } else {
                                phi_1422_ = f32();
                            }
                            let _e617_ = phi_1422_;
                            phi_1424_ = _e617_;
                            phi_1425_ = select(true, false, _e615_);
                        }
                        let _e620_ = phi_1424_;
                        let _e622_ = phi_1425_;
                        phi_1504_ = select(_e620_, 0f, _e622_);
                    } else {
                        if (_e597_ < 0f) {
                            phi_1503_ = 0f;
                        } else {
                            if (_e597_ > 128f) {
                                phi_1502_ = 128f;
                            } else {
                                if _e600_ {
                                    phi_1496_ = true;
                                } else {
                                    phi_1496_ = (_e599_ == 0u);
                                }
                                let _e605_ = phi_1496_;
                                phi_1502_ = bitcast<f32>(select(_e598_, 0u, _e605_));
                            }
                            let _e609_ = phi_1502_;
                            phi_1503_ = _e609_;
                        }
                        let _e611_ = phi_1503_;
                        phi_1504_ = _e611_;
                    }
                    let _e625_ = phi_1504_;
                    let _e626_ = (_e239_ + 2u);
                    if _e240_ {
                        phi_1516_ = 0u;
                    } else {
                        let _e627_ = (_e121_ - 1u);
                        phi_1516_ = select(_e627_, _e626_, (_e626_ < _e627_));
                    }
                    let _e631_ = phi_1516_;
                    if (_e631_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e635_ = seed_params.member[_e631_];
                    let _e636_ = bitcast<u32>(_e635_);
                    let _e637_ = (_e636_ & 2139095040u);
                    if (_e637_ == 2139095040u) {
                        phi_1534_ = true;
                    } else {
                        phi_1534_ = (_e637_ == 0u);
                    }
                    let _e641_ = phi_1534_;
                    let _e643_ = bitcast<f32>(select(_e636_, 0u, _e641_));
                    let _e644_ = (_e239_ + 3u);
                    if _e240_ {
                        phi_1551_ = 0u;
                    } else {
                        let _e645_ = (_e121_ - 1u);
                        phi_1551_ = select(_e645_, _e644_, (_e644_ < _e645_));
                    }
                    let _e649_ = phi_1551_;
                    if (_e649_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e653_ = seed_params.member[_e649_];
                    let _e654_ = bitcast<u32>(_e653_);
                    let _e655_ = (_e654_ & 2139095040u);
                    if (_e655_ == 2139095040u) {
                        phi_1569_ = true;
                    } else {
                        phi_1569_ = (_e655_ == 0u);
                    }
                    let _e659_ = phi_1569_;
                    let _e661_ = bitcast<f32>(select(_e654_, 0u, _e659_));
                    let _e662_ = (_e239_ + 4u);
                    if _e240_ {
                        phi_1586_ = 0u;
                    } else {
                        let _e663_ = (_e121_ - 1u);
                        phi_1586_ = select(_e663_, _e662_, (_e662_ < _e663_));
                    }
                    let _e667_ = phi_1586_;
                    if (_e667_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e671_ = seed_params.member[_e667_];
                    let _e672_ = bitcast<u32>(_e671_);
                    let _e673_ = (_e672_ & 2139095040u);
                    if (_e673_ == 2139095040u) {
                        phi_1604_ = true;
                    } else {
                        phi_1604_ = (_e673_ == 0u);
                    }
                    let _e677_ = phi_1604_;
                    let _e679_ = bitcast<f32>(select(_e672_, 0u, _e677_));
                    let _e680_ = (_e239_ + 5u);
                    if _e240_ {
                        phi_1621_ = 0u;
                    } else {
                        let _e681_ = (_e121_ - 1u);
                        phi_1621_ = select(_e681_, _e680_, (_e680_ < _e681_));
                    }
                    let _e685_ = phi_1621_;
                    if (_e685_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e689_ = seed_params.member[_e685_];
                    let _e690_ = bitcast<u32>(_e689_);
                    let _e691_ = (_e690_ & 2139095040u);
                    if (_e691_ == 2139095040u) {
                        phi_1639_ = true;
                    } else {
                        phi_1639_ = (_e691_ == 0u);
                    }
                    let _e695_ = phi_1639_;
                    let _e697_ = bitcast<f32>(select(_e690_, 0u, _e695_));
                    let _e698_ = (_e239_ + 12u);
                    if _e240_ {
                        phi_1656_ = 0u;
                    } else {
                        let _e699_ = (_e121_ - 1u);
                        phi_1656_ = select(_e699_, _e698_, (_e698_ < _e699_));
                    }
                    let _e703_ = phi_1656_;
                    if (_e703_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e707_ = seed_params.member[_e703_];
                    let _e708_ = bitcast<u32>(_e707_);
                    let _e709_ = (_e708_ & 2139095040u);
                    let _e710_ = (_e709_ == 2139095040u);
                    if _e710_ {
                        if ((_e708_ & 2147483647u) > 2139095040u) {
                            phi_1708_ = f32();
                            phi_1709_ = true;
                        } else {
                            let _e725_ = ((_e708_ & 2147483648u) == 0u);
                            if _e725_ {
                                phi_1706_ = 1f;
                            } else {
                                phi_1706_ = f32();
                            }
                            let _e727_ = phi_1706_;
                            phi_1708_ = _e727_;
                            phi_1709_ = select(true, false, _e725_);
                        }
                        let _e730_ = phi_1708_;
                        let _e732_ = phi_1709_;
                        if _e732_ {
                            phi_1729_ = -1f;
                        } else {
                            phi_1729_ = _e730_;
                        }
                        let _e734_ = phi_1729_;
                        phi_1788_ = _e734_;
                    } else {
                        if (_e707_ < -1f) {
                            phi_1787_ = -1f;
                        } else {
                            if (_e707_ > 1f) {
                                phi_1786_ = 1f;
                            } else {
                                if _e710_ {
                                    phi_1780_ = true;
                                } else {
                                    phi_1780_ = (_e709_ == 0u);
                                }
                                let _e715_ = phi_1780_;
                                phi_1786_ = bitcast<f32>(select(_e708_, 0u, _e715_));
                            }
                            let _e719_ = phi_1786_;
                            phi_1787_ = _e719_;
                        }
                        let _e721_ = phi_1787_;
                        phi_1788_ = _e721_;
                    }
                    let _e736_ = phi_1788_;
                    let _e737_ = (_e239_ + 13u);
                    if _e240_ {
                        phi_1800_ = 0u;
                    } else {
                        let _e738_ = (_e121_ - 1u);
                        phi_1800_ = select(_e738_, _e737_, (_e737_ < _e738_));
                    }
                    let _e742_ = phi_1800_;
                    if (_e742_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e746_ = seed_params.member[_e742_];
                    let _e747_ = bitcast<u32>(_e746_);
                    let _e748_ = (_e747_ & 2139095040u);
                    let _e749_ = (_e748_ == 2139095040u);
                    if _e749_ {
                        if ((_e747_ & 2147483647u) > 2139095040u) {
                            phi_1852_ = f32();
                            phi_1853_ = true;
                        } else {
                            let _e764_ = ((_e747_ & 2147483648u) == 0u);
                            if _e764_ {
                                phi_1850_ = 4f;
                            } else {
                                phi_1850_ = f32();
                            }
                            let _e766_ = phi_1850_;
                            phi_1852_ = _e766_;
                            phi_1853_ = select(true, false, _e764_);
                        }
                        let _e769_ = phi_1852_;
                        let _e771_ = phi_1853_;
                        if _e771_ {
                            phi_1873_ = -4f;
                        } else {
                            phi_1873_ = _e769_;
                        }
                        let _e773_ = phi_1873_;
                        phi_1932_ = _e773_;
                    } else {
                        if (_e746_ < -4f) {
                            phi_1931_ = -4f;
                        } else {
                            if (_e746_ > 4f) {
                                phi_1930_ = 4f;
                            } else {
                                if _e749_ {
                                    phi_1924_ = true;
                                } else {
                                    phi_1924_ = (_e748_ == 0u);
                                }
                                let _e754_ = phi_1924_;
                                phi_1930_ = bitcast<f32>(select(_e747_, 0u, _e754_));
                            }
                            let _e758_ = phi_1930_;
                            phi_1931_ = _e758_;
                        }
                        let _e760_ = phi_1931_;
                        phi_1932_ = _e760_;
                    }
                    let _e775_ = phi_1932_;
                    let _e776_ = (_e239_ + 14u);
                    if _e240_ {
                        phi_1944_ = 0u;
                    } else {
                        let _e777_ = (_e121_ - 1u);
                        phi_1944_ = select(_e777_, _e776_, (_e776_ < _e777_));
                    }
                    let _e781_ = phi_1944_;
                    if (_e781_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e785_ = seed_params.member[_e781_];
                    let _e786_ = bitcast<u32>(_e785_);
                    let _e787_ = (_e786_ & 2139095040u);
                    let _e788_ = (_e787_ == 2139095040u);
                    if _e788_ {
                        if ((_e786_ & 2147483647u) > 2139095040u) {
                            phi_1996_ = f32();
                            phi_1997_ = true;
                        } else {
                            let _e803_ = ((_e786_ & 2147483648u) == 0u);
                            if _e803_ {
                                phi_1994_ = 64f;
                            } else {
                                phi_1994_ = f32();
                            }
                            let _e805_ = phi_1994_;
                            phi_1996_ = _e805_;
                            phi_1997_ = select(true, false, _e803_);
                        }
                        let _e808_ = phi_1996_;
                        let _e810_ = phi_1997_;
                        if _e810_ {
                            phi_2017_ = -64f;
                        } else {
                            phi_2017_ = _e808_;
                        }
                        let _e812_ = phi_2017_;
                        phi_2076_ = _e812_;
                    } else {
                        if (_e785_ < -64f) {
                            phi_2075_ = -64f;
                        } else {
                            if (_e785_ > 64f) {
                                phi_2074_ = 64f;
                            } else {
                                if _e788_ {
                                    phi_2068_ = true;
                                } else {
                                    phi_2068_ = (_e787_ == 0u);
                                }
                                let _e793_ = phi_2068_;
                                phi_2074_ = bitcast<f32>(select(_e786_, 0u, _e793_));
                            }
                            let _e797_ = phi_2074_;
                            phi_2075_ = _e797_;
                        }
                        let _e799_ = phi_2075_;
                        phi_2076_ = _e799_;
                    }
                    let _e814_ = phi_2076_;
                    let _e815_ = (_e239_ + 15u);
                    if _e240_ {
                        phi_2088_ = 0u;
                    } else {
                        let _e816_ = (_e121_ - 1u);
                        phi_2088_ = select(_e816_, _e815_, (_e815_ < _e816_));
                    }
                    let _e820_ = phi_2088_;
                    if (_e820_ < _e121_) {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e824_ = seed_params.member[_e820_];
                    let _e825_ = bitcast<u32>(_e824_);
                    let _e826_ = (_e825_ & 2139095040u);
                    let _e827_ = (_e826_ == 2139095040u);
                    if _e827_ {
                        if ((_e825_ & 2147483647u) > 2139095040u) {
                            phi_2140_ = f32();
                            phi_2141_ = true;
                        } else {
                            let _e842_ = ((_e825_ & 2147483648u) == 0u);
                            if _e842_ {
                                phi_2138_ = 64f;
                            } else {
                                phi_2138_ = f32();
                            }
                            let _e844_ = phi_2138_;
                            phi_2140_ = _e844_;
                            phi_2141_ = select(true, false, _e842_);
                        }
                        let _e847_ = phi_2140_;
                        let _e849_ = phi_2141_;
                        if _e849_ {
                            phi_2161_ = -64f;
                        } else {
                            phi_2161_ = _e847_;
                        }
                        let _e851_ = phi_2161_;
                        phi_2220_ = _e851_;
                    } else {
                        if (_e824_ < -64f) {
                            phi_2219_ = -64f;
                        } else {
                            if (_e824_ > 64f) {
                                phi_2218_ = 64f;
                            } else {
                                if _e827_ {
                                    phi_2212_ = true;
                                } else {
                                    phi_2212_ = (_e826_ == 0u);
                                }
                                let _e832_ = phi_2212_;
                                phi_2218_ = bitcast<f32>(select(_e825_, 0u, _e832_));
                            }
                            let _e836_ = phi_2218_;
                            phi_2219_ = _e836_;
                        }
                        let _e838_ = phi_2219_;
                        phi_2220_ = _e838_;
                    }
                    let _e853_ = phi_2220_;
                    if _e160_ {
                        let _e855_ = (_e595_ + _e595_);
                        let _e856_ = (_e596_ + _e596_);
                        phi_2242_ = type_23_(((_e594_ + _e594_) - 1f), -(_e855_), _e855_, (1f - _e856_), (_e855_ * _e196_.x), (_e856_ * _e196_.x));
                    } else {
                        phi_2242_ = type_23_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e865_ = phi_2242_;
                    let _e867_ = select(255u, _e114_, (_e114_ < 255u));
                    let _e869_ = (_e867_ < 1536u);
                    if _e869_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e865_.member;
                    let _e872_ = (256u + _e867_);
                    let _e873_ = (_e872_ < 1536u);
                    if _e873_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e865_.member_1_;
                    let _e876_ = (512u + _e867_);
                    let _e877_ = (_e876_ < 1536u);
                    if _e877_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e876_] = _e865_.member_2_;
                    let _e880_ = (768u + _e867_);
                    let _e881_ = (_e880_ < 1536u);
                    if _e881_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e880_] = _e865_.member_3_;
                    let _e884_ = (1024u + _e867_);
                    let _e885_ = (_e884_ < 1536u);
                    if _e885_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e884_] = _e865_.member_4_;
                    let _e888_ = (1280u + _e867_);
                    let _e889_ = (_e888_ < 1536u);
                    if _e889_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e888_] = _e865_.member_5_;
                    workgroupBarrier();
                    phi_2302_ = 1u;
                    loop {
                        let _e892_ = phi_2302_;
                        let _e893_ = (_e892_ < 256u);
                        if _e893_ {
                            let _e895_ = (_e892_ << bitcast<u32>(1i));
                            let _e898_ = (((_e114_ + 1u) * _e895_) - 1u);
                            if (_e898_ < 256u) {
                                let _e900_ = (_e898_ - _e892_);
                                let _e902_ = select(255u, _e900_, (_e900_ < 255u));
                                if (_e902_ < 1536u) {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e905_ = scan_scratch[_e902_];
                                let _e906_ = (256u + _e902_);
                                if (_e906_ < 1536u) {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e909_ = scan_scratch[_e906_];
                                let _e910_ = (512u + _e902_);
                                if (_e910_ < 1536u) {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e913_ = scan_scratch[_e910_];
                                let _e914_ = (768u + _e902_);
                                if (_e914_ < 1536u) {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e917_ = scan_scratch[_e914_];
                                let _e918_ = (1024u + _e902_);
                                if (_e918_ < 1536u) {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e921_ = scan_scratch[_e918_];
                                let _e922_ = (1280u + _e902_);
                                if (_e922_ < 1536u) {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e925_ = scan_scratch[_e922_];
                                let _e927_ = select(255u, _e898_, (_e898_ < 255u));
                                let _e928_ = (_e927_ < 1536u);
                                if _e928_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e930_ = scan_scratch[_e927_];
                                let _e931_ = (256u + _e927_);
                                let _e932_ = (_e931_ < 1536u);
                                if _e932_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e934_ = scan_scratch[_e931_];
                                let _e935_ = (512u + _e927_);
                                let _e936_ = (_e935_ < 1536u);
                                if _e936_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e938_ = scan_scratch[_e935_];
                                let _e939_ = (768u + _e927_);
                                let _e940_ = (_e939_ < 1536u);
                                if _e940_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e942_ = scan_scratch[_e939_];
                                let _e943_ = (1024u + _e927_);
                                let _e944_ = (_e943_ < 1536u);
                                if _e944_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e946_ = scan_scratch[_e943_];
                                let _e947_ = (1280u + _e927_);
                                let _e948_ = (_e947_ < 1536u);
                                if _e948_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                let _e950_ = scan_scratch[_e947_];
                                if _e928_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                scan_scratch[_e927_] = ((_e930_ * _e905_) + (_e934_ * _e913_));
                                if _e932_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                scan_scratch[_e931_] = ((_e930_ * _e909_) + (_e934_ * _e917_));
                                if _e936_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                scan_scratch[_e935_] = ((_e938_ * _e905_) + (_e942_ * _e913_));
                                if _e940_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                scan_scratch[_e939_] = ((_e938_ * _e909_) + (_e942_ * _e917_));
                                if _e944_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                scan_scratch[_e943_] = (((_e930_ * _e921_) + (_e934_ * _e925_)) + _e946_);
                                if _e948_ {
                                } else {
                                    phi_11818_ = true;
                                    break;
                                }
                                scan_scratch[_e947_] = (((_e938_ * _e921_) + (_e942_ * _e925_)) + _e950_);
                            }
                            workgroupBarrier();
                            phi_2303_ = _e895_;
                        } else {
                            phi_2303_ = u32();
                        }
                        let _e972_ = phi_2303_;
                        phi_2302_ = _e972_;
                        phi_11818_ = _e199_;
                        if !(_e893_) {
                            break;
                        }
                    }
                    let _e975_ = phi_11818_;
                    phi_12454_ = _e975_;
                    if _e975_ {
                        break;
                    }
                    let _e976_ = (_e114_ == 255u);
                    if _e976_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_2565_ = 128u;
                    loop {
                        let _e984_ = phi_2565_;
                        let _e985_ = (_e984_ > 0u);
                        if _e985_ {
                            let _e990_ = (((_e114_ + 1u) * (_e984_ << bitcast<u32>(1i))) - 1u);
                            if (_e990_ < 256u) {
                                let _e992_ = (_e990_ - _e984_);
                                let _e994_ = select(255u, _e992_, (_e992_ < 255u));
                                let _e995_ = (_e994_ < 1536u);
                                if _e995_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e997_ = scan_scratch[_e994_];
                                let _e998_ = (256u + _e994_);
                                let _e999_ = (_e998_ < 1536u);
                                if _e999_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1001_ = scan_scratch[_e998_];
                                let _e1002_ = (512u + _e994_);
                                let _e1003_ = (_e1002_ < 1536u);
                                if _e1003_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1005_ = scan_scratch[_e1002_];
                                let _e1006_ = (768u + _e994_);
                                let _e1007_ = (_e1006_ < 1536u);
                                if _e1007_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1009_ = scan_scratch[_e1006_];
                                let _e1010_ = (1024u + _e994_);
                                let _e1011_ = (_e1010_ < 1536u);
                                if _e1011_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1013_ = scan_scratch[_e1010_];
                                let _e1014_ = (1280u + _e994_);
                                let _e1015_ = (_e1014_ < 1536u);
                                if _e1015_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1017_ = scan_scratch[_e1014_];
                                let _e1019_ = select(255u, _e990_, (_e990_ < 255u));
                                let _e1020_ = (_e1019_ < 1536u);
                                if _e1020_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1022_ = scan_scratch[_e1019_];
                                let _e1023_ = (256u + _e1019_);
                                let _e1024_ = (_e1023_ < 1536u);
                                if _e1024_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1026_ = scan_scratch[_e1023_];
                                let _e1027_ = (512u + _e1019_);
                                let _e1028_ = (_e1027_ < 1536u);
                                if _e1028_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1030_ = scan_scratch[_e1027_];
                                let _e1031_ = (768u + _e1019_);
                                let _e1032_ = (_e1031_ < 1536u);
                                if _e1032_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1034_ = scan_scratch[_e1031_];
                                let _e1035_ = (1024u + _e1019_);
                                let _e1036_ = (_e1035_ < 1536u);
                                if _e1036_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1038_ = scan_scratch[_e1035_];
                                let _e1039_ = (1280u + _e1019_);
                                let _e1040_ = (_e1039_ < 1536u);
                                if _e1040_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                let _e1042_ = scan_scratch[_e1039_];
                                if _e995_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e994_] = _e1022_;
                                if _e999_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e998_] = _e1026_;
                                if _e1003_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1002_] = _e1030_;
                                if _e1007_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1006_] = _e1034_;
                                if _e1011_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1010_] = _e1038_;
                                if _e1015_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1014_] = _e1042_;
                                if _e1020_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1019_] = ((_e997_ * _e1022_) + (_e1001_ * _e1030_));
                                if _e1024_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1023_] = ((_e997_ * _e1026_) + (_e1001_ * _e1034_));
                                if _e1028_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1027_] = ((_e1005_ * _e1022_) + (_e1009_ * _e1030_));
                                if _e1032_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1031_] = ((_e1005_ * _e1026_) + (_e1009_ * _e1034_));
                                if _e1036_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1035_] = (((_e997_ * _e1038_) + (_e1001_ * _e1042_)) + _e1013_);
                                if _e1040_ {
                                } else {
                                    phi_12119_ = true;
                                    break;
                                }
                                scan_scratch[_e1039_] = (((_e1005_ * _e1038_) + (_e1009_ * _e1042_)) + _e1017_);
                            }
                            workgroupBarrier();
                            phi_2566_ = (_e984_ >> bitcast<u32>(1i));
                        } else {
                            phi_2566_ = u32();
                        }
                        let _e1066_ = phi_2566_;
                        phi_2565_ = _e1066_;
                        phi_12119_ = _e975_;
                        if !(_e985_) {
                            break;
                        }
                    }
                    let _e1069_ = phi_12119_;
                    phi_12454_ = _e1069_;
                    if _e1069_ {
                        break;
                    }
                    if _e869_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e873_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e877_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e881_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e885_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e1070_ = scan_scratch[_e884_];
                    if _e889_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e1071_ = scan_scratch[_e888_];
                    workgroupBarrier();
                    if _e160_ {
                        let _e1072_ = (_e196_.x - _e1071_);
                        let _e1075_ = ((_e594_ * _e1070_) + (_e595_ * _e1072_));
                        let _e1079_ = ((_e1071_ + (_e595_ * _e1070_)) + (_e596_ * _e1072_));
                        let _e1090_ = bitcast<u32>(((((_e643_ * _e1079_) + (_e661_ * _e1075_)) + (_e679_ * ((_e196_.x - (_e580_ * _e1075_)) - _e1079_))) + (_e697_ * _e196_.x)));
                        let _e1091_ = (_e1090_ & 2139095040u);
                        if (_e1091_ == 2139095040u) {
                            phi_3108_ = true;
                        } else {
                            phi_3108_ = (_e1091_ == 0u);
                        }
                        let _e1095_ = phi_3108_;
                        let _e1098_ = bitcast<u32>(_e1075_);
                        let _e1099_ = (_e1098_ & 2139095040u);
                        if (_e1099_ == 2139095040u) {
                            phi_3124_ = true;
                        } else {
                            phi_3124_ = (_e1099_ == 0u);
                        }
                        let _e1103_ = phi_3124_;
                        phi_3132_ = type_20_(bitcast<f32>(select(_e1090_, 0u, _e1095_)), bitcast<f32>(select(_e1098_, 0u, _e1103_)));
                    } else {
                        phi_3132_ = type_20_(0f, 0f);
                    }
                    let _e1108_ = phi_3132_;
                    if _e160_ {
                        let _e1112_ = (_e595_ + _e595_);
                        let _e1113_ = (_e596_ + _e596_);
                        phi_3155_ = type_23_(((_e594_ + _e594_) - 1f), -(_e1112_), _e1112_, (1f - _e1113_), (_e1112_ * _e196_.y), (_e1113_ * _e196_.y));
                    } else {
                        phi_3155_ = type_23_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e1122_ = phi_3155_;
                    if _e869_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e1122_.member;
                    if _e873_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e1122_.member_1_;
                    if _e877_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e876_] = _e1122_.member_2_;
                    if _e881_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e880_] = _e1122_.member_3_;
                    if _e885_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e884_] = _e1122_.member_4_;
                    if _e889_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e888_] = _e1122_.member_5_;
                    workgroupBarrier();
                    phi_3215_ = 1u;
                    loop {
                        let _e1130_ = phi_3215_;
                        let _e1131_ = (_e1130_ < 256u);
                        if _e1131_ {
                            let _e1133_ = (_e1130_ << bitcast<u32>(1i));
                            let _e1136_ = (((_e114_ + 1u) * _e1133_) - 1u);
                            if (_e1136_ < 256u) {
                                let _e1138_ = (_e1136_ - _e1130_);
                                let _e1140_ = select(255u, _e1138_, (_e1138_ < 255u));
                                if (_e1140_ < 1536u) {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1143_ = scan_scratch[_e1140_];
                                let _e1144_ = (256u + _e1140_);
                                if (_e1144_ < 1536u) {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1147_ = scan_scratch[_e1144_];
                                let _e1148_ = (512u + _e1140_);
                                if (_e1148_ < 1536u) {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1151_ = scan_scratch[_e1148_];
                                let _e1152_ = (768u + _e1140_);
                                if (_e1152_ < 1536u) {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1155_ = scan_scratch[_e1152_];
                                let _e1156_ = (1024u + _e1140_);
                                if (_e1156_ < 1536u) {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1159_ = scan_scratch[_e1156_];
                                let _e1160_ = (1280u + _e1140_);
                                if (_e1160_ < 1536u) {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1163_ = scan_scratch[_e1160_];
                                let _e1165_ = select(255u, _e1136_, (_e1136_ < 255u));
                                let _e1166_ = (_e1165_ < 1536u);
                                if _e1166_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1168_ = scan_scratch[_e1165_];
                                let _e1169_ = (256u + _e1165_);
                                let _e1170_ = (_e1169_ < 1536u);
                                if _e1170_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1172_ = scan_scratch[_e1169_];
                                let _e1173_ = (512u + _e1165_);
                                let _e1174_ = (_e1173_ < 1536u);
                                if _e1174_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1176_ = scan_scratch[_e1173_];
                                let _e1177_ = (768u + _e1165_);
                                let _e1178_ = (_e1177_ < 1536u);
                                if _e1178_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1180_ = scan_scratch[_e1177_];
                                let _e1181_ = (1024u + _e1165_);
                                let _e1182_ = (_e1181_ < 1536u);
                                if _e1182_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1184_ = scan_scratch[_e1181_];
                                let _e1185_ = (1280u + _e1165_);
                                let _e1186_ = (_e1185_ < 1536u);
                                if _e1186_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                let _e1188_ = scan_scratch[_e1185_];
                                if _e1166_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                scan_scratch[_e1165_] = ((_e1168_ * _e1143_) + (_e1172_ * _e1151_));
                                if _e1170_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                scan_scratch[_e1169_] = ((_e1168_ * _e1147_) + (_e1172_ * _e1155_));
                                if _e1174_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                scan_scratch[_e1173_] = ((_e1176_ * _e1143_) + (_e1180_ * _e1151_));
                                if _e1178_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                scan_scratch[_e1177_] = ((_e1176_ * _e1147_) + (_e1180_ * _e1155_));
                                if _e1182_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                scan_scratch[_e1181_] = (((_e1168_ * _e1159_) + (_e1172_ * _e1163_)) + _e1184_);
                                if _e1186_ {
                                } else {
                                    phi_12134_ = true;
                                    break;
                                }
                                scan_scratch[_e1185_] = (((_e1176_ * _e1159_) + (_e1180_ * _e1163_)) + _e1188_);
                            }
                            workgroupBarrier();
                            phi_3216_ = _e1133_;
                        } else {
                            phi_3216_ = u32();
                        }
                        let _e1210_ = phi_3216_;
                        phi_3215_ = _e1210_;
                        phi_12134_ = _e1069_;
                        if !(_e1131_) {
                            break;
                        }
                    }
                    let _e1213_ = phi_12134_;
                    phi_12454_ = _e1213_;
                    if _e1213_ {
                        break;
                    }
                    if _e976_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_3478_ = 128u;
                    loop {
                        let _e1221_ = phi_3478_;
                        let _e1222_ = (_e1221_ > 0u);
                        if _e1222_ {
                            let _e1227_ = (((_e114_ + 1u) * (_e1221_ << bitcast<u32>(1i))) - 1u);
                            if (_e1227_ < 256u) {
                                let _e1229_ = (_e1227_ - _e1221_);
                                let _e1231_ = select(255u, _e1229_, (_e1229_ < 255u));
                                let _e1232_ = (_e1231_ < 1536u);
                                if _e1232_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1234_ = scan_scratch[_e1231_];
                                let _e1235_ = (256u + _e1231_);
                                let _e1236_ = (_e1235_ < 1536u);
                                if _e1236_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1238_ = scan_scratch[_e1235_];
                                let _e1239_ = (512u + _e1231_);
                                let _e1240_ = (_e1239_ < 1536u);
                                if _e1240_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1242_ = scan_scratch[_e1239_];
                                let _e1243_ = (768u + _e1231_);
                                let _e1244_ = (_e1243_ < 1536u);
                                if _e1244_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1246_ = scan_scratch[_e1243_];
                                let _e1247_ = (1024u + _e1231_);
                                let _e1248_ = (_e1247_ < 1536u);
                                if _e1248_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1250_ = scan_scratch[_e1247_];
                                let _e1251_ = (1280u + _e1231_);
                                let _e1252_ = (_e1251_ < 1536u);
                                if _e1252_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1254_ = scan_scratch[_e1251_];
                                let _e1256_ = select(255u, _e1227_, (_e1227_ < 255u));
                                let _e1257_ = (_e1256_ < 1536u);
                                if _e1257_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1259_ = scan_scratch[_e1256_];
                                let _e1260_ = (256u + _e1256_);
                                let _e1261_ = (_e1260_ < 1536u);
                                if _e1261_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1263_ = scan_scratch[_e1260_];
                                let _e1264_ = (512u + _e1256_);
                                let _e1265_ = (_e1264_ < 1536u);
                                if _e1265_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1267_ = scan_scratch[_e1264_];
                                let _e1268_ = (768u + _e1256_);
                                let _e1269_ = (_e1268_ < 1536u);
                                if _e1269_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1271_ = scan_scratch[_e1268_];
                                let _e1272_ = (1024u + _e1256_);
                                let _e1273_ = (_e1272_ < 1536u);
                                if _e1273_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1275_ = scan_scratch[_e1272_];
                                let _e1276_ = (1280u + _e1256_);
                                let _e1277_ = (_e1276_ < 1536u);
                                if _e1277_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                let _e1279_ = scan_scratch[_e1276_];
                                if _e1232_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1231_] = _e1259_;
                                if _e1236_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1235_] = _e1263_;
                                if _e1240_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1239_] = _e1267_;
                                if _e1244_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1243_] = _e1271_;
                                if _e1248_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1247_] = _e1275_;
                                if _e1252_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1251_] = _e1279_;
                                if _e1257_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1256_] = ((_e1234_ * _e1259_) + (_e1238_ * _e1267_));
                                if _e1261_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1260_] = ((_e1234_ * _e1263_) + (_e1238_ * _e1271_));
                                if _e1265_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1264_] = ((_e1242_ * _e1259_) + (_e1246_ * _e1267_));
                                if _e1269_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1268_] = ((_e1242_ * _e1263_) + (_e1246_ * _e1271_));
                                if _e1273_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1272_] = (((_e1234_ * _e1275_) + (_e1238_ * _e1279_)) + _e1250_);
                                if _e1277_ {
                                } else {
                                    phi_12154_ = true;
                                    break;
                                }
                                scan_scratch[_e1276_] = (((_e1242_ * _e1275_) + (_e1246_ * _e1279_)) + _e1254_);
                            }
                            workgroupBarrier();
                            phi_3479_ = (_e1221_ >> bitcast<u32>(1i));
                        } else {
                            phi_3479_ = u32();
                        }
                        let _e1303_ = phi_3479_;
                        phi_3478_ = _e1303_;
                        phi_12154_ = _e1213_;
                        if !(_e1222_) {
                            break;
                        }
                    }
                    let _e1306_ = phi_12154_;
                    phi_12454_ = _e1306_;
                    if _e1306_ {
                        break;
                    }
                    if _e869_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e873_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e877_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e881_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    if _e885_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e1307_ = scan_scratch[_e884_];
                    if _e889_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    let _e1308_ = scan_scratch[_e888_];
                    workgroupBarrier();
                    if _e160_ {
                        let _e1309_ = (_e196_.y - _e1308_);
                        let _e1312_ = ((_e594_ * _e1307_) + (_e595_ * _e1309_));
                        let _e1316_ = ((_e1308_ + (_e595_ * _e1307_)) + (_e596_ * _e1309_));
                        let _e1327_ = bitcast<u32>(((((_e643_ * _e1316_) + (_e661_ * _e1312_)) + (_e679_ * ((_e196_.y - (_e580_ * _e1312_)) - _e1316_))) + (_e697_ * _e196_.y)));
                        let _e1328_ = (_e1327_ & 2139095040u);
                        if (_e1328_ == 2139095040u) {
                            phi_4021_ = true;
                        } else {
                            phi_4021_ = (_e1328_ == 0u);
                        }
                        let _e1332_ = phi_4021_;
                        let _e1335_ = bitcast<u32>(_e1312_);
                        let _e1336_ = (_e1335_ & 2139095040u);
                        if (_e1336_ == 2139095040u) {
                            phi_4037_ = true;
                        } else {
                            phi_4037_ = (_e1336_ == 0u);
                        }
                        let _e1340_ = phi_4037_;
                        phi_4045_ = type_20_(bitcast<f32>(select(_e1327_, 0u, _e1332_)), bitcast<f32>(select(_e1335_, 0u, _e1340_)));
                    } else {
                        phi_4045_ = type_20_(0f, 0f);
                    }
                    let _e1345_ = phi_4045_;
                    if _e869_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e1108_.member;
                    if _e873_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e1345_.member;
                    if _e877_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e876_] = _e1108_.member_1_;
                    if _e881_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e880_] = _e1345_.member_1_;
                    if _e885_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e884_] = 0f;
                    if _e889_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e888_] = 0f;
                    workgroupBarrier();
                    if (_e114_ == 0u) {
                        phi_4110_ = type_15_(0u, select(_e156_, 256u, (256u < _e156_)));
                        phi_4113_ = type_22_(array<f32, 4>(0f, 0f, 0f, 0f));
                        phi_4115_ = type_22_(array<f32, 4>(0f, 0f, 0f, 0f));
                        loop {
                            let _e1351_ = phi_4110_;
                            let _e1353_ = phi_4113_;
                            let _e1355_ = phi_4115_;
                            if (_e1351_.member < _e1351_.member_1_) {
                                phi_4111_ = type_15_((_e1351_.member + 1u), _e1351_.member_1_);
                                phi_4138_ = type_15_(1u, _e1351_.member);
                            } else {
                                phi_4111_ = _e1351_;
                                phi_4138_ = type_15_(0u, u32());
                            }
                            let _e1368_ = phi_4111_;
                            let _e1370_ = phi_4138_;
                            let _e1374_ = (bitcast<i32>(_e1370_.member) != 0i);
                            if _e1374_ {
                                if (_e1370_.member_1_ < 1536u) {
                                } else {
                                    phi_12169_ = true;
                                    break;
                                }
                                let _e1377_ = scan_scratch[_e1370_.member_1_];
                                let _e1378_ = (256u + _e1370_.member_1_);
                                if (_e1378_ < 1536u) {
                                } else {
                                    phi_12169_ = true;
                                    break;
                                }
                                let _e1381_ = scan_scratch[_e1378_];
                                let _e1382_ = (512u + _e1370_.member_1_);
                                if (_e1382_ < 1536u) {
                                } else {
                                    phi_12169_ = true;
                                    break;
                                }
                                let _e1385_ = scan_scratch[_e1382_];
                                let _e1386_ = (768u + _e1370_.member_1_);
                                if (_e1386_ < 1536u) {
                                } else {
                                    phi_12169_ = true;
                                    break;
                                }
                                let _e1389_ = scan_scratch[_e1386_];
                                let _e1390_ = (_e736_ * _e1377_);
                                let _e1391_ = (_e736_ * _e1381_);
                                let _e1394_ = bitcast<u32>((_e625_ * _e1385_));
                                let _e1395_ = (_e1394_ & 2139095040u);
                                let _e1396_ = (_e1395_ == 2139095040u);
                                if _e1396_ {
                                    phi_4191_ = true;
                                } else {
                                    phi_4191_ = (_e1395_ == 0u);
                                }
                                let _e1399_ = phi_4191_;
                                if _e1396_ {
                                    phi_4207_ = true;
                                } else {
                                    phi_4207_ = (_e1395_ == 0u);
                                }
                                let _e1404_ = phi_4207_;
                                if _e1396_ {
                                    phi_4223_ = true;
                                } else {
                                    phi_4223_ = (_e1395_ == 0u);
                                }
                                let _e1409_ = phi_4223_;
                                if _e1396_ {
                                    phi_4239_ = true;
                                } else {
                                    phi_4239_ = (_e1395_ == 0u);
                                }
                                let _e1414_ = phi_4239_;
                                let _e1433_ = (_e558_ <= 0f);
                                let _e1438_ = (1f - select(select(select(0f, 1f, (1f <= _e462_)), _e462_, (_e462_ <= 1f)), 0f, (_e462_ <= 0f)));
                                let _e1439_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1394_, 0u, _e1399_))) & 2147483647u))), 0f, _e1433_));
                                let _e1440_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1394_, 0u, _e1404_))) & 2147483647u))), 0f, _e1433_));
                                let _e1441_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1394_, 0u, _e1409_))) & 2147483647u))), 0f, _e1433_));
                                let _e1442_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1394_, 0u, _e1414_))) & 2147483647u))), 0f, _e1433_));
                                if (_e1439_ != _e1439_) {
                                    phi_8438_ = true;
                                } else {
                                    phi_8438_ = (0f >= _e1439_);
                                }
                                let _e1446_ = phi_8438_;
                                let _e1447_ = select(_e1439_, 0f, _e1446_);
                                if (_e1440_ != _e1440_) {
                                    phi_8453_ = true;
                                } else {
                                    phi_8453_ = (0f >= _e1440_);
                                }
                                let _e1451_ = phi_8453_;
                                let _e1452_ = select(_e1440_, 0f, _e1451_);
                                if (_e1441_ != _e1441_) {
                                    phi_8468_ = true;
                                } else {
                                    phi_8468_ = (0f >= _e1441_);
                                }
                                let _e1456_ = phi_8468_;
                                let _e1457_ = select(_e1441_, 0f, _e1456_);
                                if (_e1442_ != _e1442_) {
                                    phi_8483_ = true;
                                } else {
                                    phi_8483_ = (0f >= _e1442_);
                                }
                                let _e1461_ = phi_8483_;
                                let _e1462_ = select(_e1442_, 0f, _e1461_);
                                if (_e1447_ != _e1447_) {
                                    phi_8498_ = true;
                                } else {
                                    phi_8498_ = (1f <= _e1447_);
                                }
                                let _e1466_ = phi_8498_;
                                let _e1467_ = select(_e1447_, 1f, _e1466_);
                                if (_e1452_ != _e1452_) {
                                    phi_8513_ = true;
                                } else {
                                    phi_8513_ = (1f <= _e1452_);
                                }
                                let _e1471_ = phi_8513_;
                                let _e1472_ = select(_e1452_, 1f, _e1471_);
                                if (_e1457_ != _e1457_) {
                                    phi_8528_ = true;
                                } else {
                                    phi_8528_ = (1f <= _e1457_);
                                }
                                let _e1476_ = phi_8528_;
                                let _e1477_ = select(_e1457_, 1f, _e1476_);
                                if (_e1462_ != _e1462_) {
                                    phi_8543_ = true;
                                } else {
                                    phi_8543_ = (1f <= _e1462_);
                                }
                                let _e1481_ = phi_8543_;
                                let _e1482_ = select(_e1462_, 1f, _e1481_);
                                let _e1485_ = (_e540_ * _e1355_.member[0]);
                                let _e1488_ = (_e540_ * _e1355_.member[1]);
                                let _e1491_ = (_e540_ * _e1355_.member[2]);
                                let _e1494_ = (_e540_ * _e1355_.member[3]);
                                let _e1495_ = (1f - _e564_);
                                let _e1496_ = (select(select(select(0f, 1f, (1f <= _e480_)), _e480_, (_e480_ <= 1f)), 0f, (_e480_ <= 0f)) * _e1495_);
                                let _e1498_ = (_e1390_ - (_e1496_ * _e1485_));
                                let _e1500_ = (_e1390_ - (_e1496_ * _e1488_));
                                let _e1502_ = (_e1390_ - (_e1496_ * _e1491_));
                                let _e1504_ = (_e1390_ - (_e1496_ * _e1494_));
                                if (_e1498_ != _e1498_) {
                                    phi_8558_ = true;
                                } else {
                                    phi_8558_ = (-1f >= _e1498_);
                                }
                                let _e1508_ = phi_8558_;
                                let _e1509_ = select(_e1498_, -1f, _e1508_);
                                if (_e1500_ != _e1500_) {
                                    phi_8573_ = true;
                                } else {
                                    phi_8573_ = (-1f >= _e1500_);
                                }
                                let _e1513_ = phi_8573_;
                                let _e1514_ = select(_e1500_, -1f, _e1513_);
                                if (_e1502_ != _e1502_) {
                                    phi_8588_ = true;
                                } else {
                                    phi_8588_ = (-1f >= _e1502_);
                                }
                                let _e1518_ = phi_8588_;
                                let _e1519_ = select(_e1502_, -1f, _e1518_);
                                if (_e1504_ != _e1504_) {
                                    phi_8603_ = true;
                                } else {
                                    phi_8603_ = (-1f >= _e1504_);
                                }
                                let _e1523_ = phi_8603_;
                                let _e1524_ = select(_e1504_, -1f, _e1523_);
                                if (_e1509_ != _e1509_) {
                                    phi_8618_ = true;
                                } else {
                                    phi_8618_ = (1f <= _e1509_);
                                }
                                let _e1528_ = phi_8618_;
                                let _e1529_ = select(_e1509_, 1f, _e1528_);
                                if (_e1514_ != _e1514_) {
                                    phi_8633_ = true;
                                } else {
                                    phi_8633_ = (1f <= _e1514_);
                                }
                                let _e1533_ = phi_8633_;
                                let _e1534_ = select(_e1514_, 1f, _e1533_);
                                if (_e1519_ != _e1519_) {
                                    phi_8648_ = true;
                                } else {
                                    phi_8648_ = (1f <= _e1519_);
                                }
                                let _e1538_ = phi_8648_;
                                let _e1539_ = select(_e1519_, 1f, _e1538_);
                                if (_e1524_ != _e1524_) {
                                    phi_8663_ = true;
                                } else {
                                    phi_8663_ = (1f <= _e1524_);
                                }
                                let _e1543_ = phi_8663_;
                                let _e1544_ = select(_e1524_, 1f, _e1543_);
                                let _e1563_ = bitcast<f32>((bitcast<u32>(_e1498_) & 2147483647u));
                                let _e1566_ = bitcast<f32>((bitcast<u32>(_e1500_) & 2147483647u));
                                let _e1569_ = bitcast<f32>((bitcast<u32>(_e1502_) & 2147483647u));
                                let _e1572_ = bitcast<f32>((bitcast<u32>(_e1504_) & 2147483647u));
                                if (_e1563_ != _e1563_) {
                                    phi_8678_ = true;
                                } else {
                                    phi_8678_ = (1000000000000000f <= _e1563_);
                                }
                                let _e1576_ = phi_8678_;
                                let _e1577_ = select(_e1563_, 1000000000000000f, _e1576_);
                                if (_e1566_ != _e1566_) {
                                    phi_8693_ = true;
                                } else {
                                    phi_8693_ = (1000000000000000f <= _e1566_);
                                }
                                let _e1581_ = phi_8693_;
                                let _e1582_ = select(_e1566_, 1000000000000000f, _e1581_);
                                if (_e1569_ != _e1569_) {
                                    phi_8708_ = true;
                                } else {
                                    phi_8708_ = (1000000000000000f <= _e1569_);
                                }
                                let _e1586_ = phi_8708_;
                                let _e1587_ = select(_e1569_, 1000000000000000f, _e1586_);
                                if (_e1572_ != _e1572_) {
                                    phi_8723_ = true;
                                } else {
                                    phi_8723_ = (1000000000000000f <= _e1572_);
                                }
                                let _e1591_ = phi_8723_;
                                let _e1592_ = select(_e1572_, 1000000000000000f, _e1591_);
                                let _e1593_ = (_e1577_ * _e1577_);
                                let _e1594_ = (_e1582_ * _e1582_);
                                let _e1595_ = (_e1587_ * _e1587_);
                                let _e1596_ = (_e1592_ * _e1592_);
                                let _e1597_ = (_e1593_ + 0.0000010000001f);
                                let _e1598_ = (_e1594_ + 0.0000010000001f);
                                let _e1599_ = (_e1595_ + 0.0000010000001f);
                                let _e1600_ = (_e1596_ + 0.0000010000001f);
                                if (_e1597_ != _e1597_) {
                                    phi_8738_ = true;
                                } else {
                                    phi_8738_ = (0.000000000001f >= _e1597_);
                                }
                                let _e1604_ = phi_8738_;
                                let _e1605_ = select(_e1597_, 0.000000000001f, _e1604_);
                                if (_e1598_ != _e1598_) {
                                    phi_8753_ = true;
                                } else {
                                    phi_8753_ = (0.000000000001f >= _e1598_);
                                }
                                let _e1609_ = phi_8753_;
                                let _e1610_ = select(_e1598_, 0.000000000001f, _e1609_);
                                if (_e1599_ != _e1599_) {
                                    phi_8768_ = true;
                                } else {
                                    phi_8768_ = (0.000000000001f >= _e1599_);
                                }
                                let _e1614_ = phi_8768_;
                                let _e1615_ = select(_e1599_, 0.000000000001f, _e1614_);
                                if (_e1600_ != _e1600_) {
                                    phi_8783_ = true;
                                } else {
                                    phi_8783_ = (0.000000000001f >= _e1600_);
                                }
                                let _e1619_ = phi_8783_;
                                let _e1620_ = select(_e1600_, 0.000000000001f, _e1619_);
                                let _e1621_ = (_e1605_ * 0.5f);
                                let _e1622_ = (_e1610_ * 0.5f);
                                let _e1623_ = (_e1615_ * 0.5f);
                                let _e1624_ = (_e1620_ * 0.5f);
                                let _e1629_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1605_) >> bitcast<u32>(1i))));
                                let _e1634_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1610_) >> bitcast<u32>(1i))));
                                let _e1639_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1615_) >> bitcast<u32>(1i))));
                                let _e1644_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1620_) >> bitcast<u32>(1i))));
                                let _e1657_ = (_e1629_ * (1.5f - (_e1621_ * (_e1629_ * _e1629_))));
                                let _e1658_ = (_e1634_ * (1.5f - (_e1622_ * (_e1634_ * _e1634_))));
                                let _e1659_ = (_e1639_ * (1.5f - (_e1623_ * (_e1639_ * _e1639_))));
                                let _e1660_ = (_e1644_ * (1.5f - (_e1624_ * (_e1644_ * _e1644_))));
                                let _e1673_ = (_e1657_ * (1.5f - (_e1621_ * (_e1657_ * _e1657_))));
                                let _e1674_ = (_e1658_ * (1.5f - (_e1622_ * (_e1658_ * _e1658_))));
                                let _e1675_ = (_e1659_ * (1.5f - (_e1623_ * (_e1659_ * _e1659_))));
                                let _e1676_ = (_e1660_ * (1.5f - (_e1624_ * (_e1660_ * _e1660_))));
                                let _e1693_ = (_e1597_ * (_e1673_ * (1.5f - (_e1621_ * (_e1673_ * _e1673_)))));
                                let _e1694_ = (_e1598_ * (_e1674_ * (1.5f - (_e1622_ * (_e1674_ * _e1674_)))));
                                let _e1695_ = (_e1599_ * (_e1675_ * (1.5f - (_e1623_ * (_e1675_ * _e1675_)))));
                                let _e1696_ = (_e1600_ * (_e1676_ * (1.5f - (_e1624_ * (_e1676_ * _e1676_)))));
                                let _e1697_ = (_e1693_ + 0.001f);
                                let _e1698_ = (_e1694_ + 0.001f);
                                let _e1699_ = (_e1695_ + 0.001f);
                                let _e1700_ = (_e1696_ + 0.001f);
                                if (_e1697_ != _e1697_) {
                                    phi_8798_ = true;
                                } else {
                                    phi_8798_ = (0.000001f >= _e1697_);
                                }
                                let _e1704_ = phi_8798_;
                                let _e1705_ = select(_e1697_, 0.000001f, _e1704_);
                                if (_e1698_ != _e1698_) {
                                    phi_8813_ = true;
                                } else {
                                    phi_8813_ = (0.000001f >= _e1698_);
                                }
                                let _e1709_ = phi_8813_;
                                let _e1710_ = select(_e1698_, 0.000001f, _e1709_);
                                if (_e1699_ != _e1699_) {
                                    phi_8828_ = true;
                                } else {
                                    phi_8828_ = (0.000001f >= _e1699_);
                                }
                                let _e1714_ = phi_8828_;
                                let _e1715_ = select(_e1699_, 0.000001f, _e1714_);
                                if (_e1700_ != _e1700_) {
                                    phi_8843_ = true;
                                } else {
                                    phi_8843_ = (0.000001f >= _e1700_);
                                }
                                let _e1719_ = phi_8843_;
                                let _e1720_ = select(_e1700_, 0.000001f, _e1719_);
                                let _e1723_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1705_)));
                                let _e1726_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1710_)));
                                let _e1729_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1715_)));
                                let _e1732_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1720_)));
                                let _e1742_ = (_e1723_ + (_e1723_ * (1f - (_e1705_ * _e1723_))));
                                let _e1744_ = (_e1726_ + (_e1726_ * (1f - (_e1710_ * _e1726_))));
                                let _e1746_ = (_e1729_ + (_e1729_ * (1f - (_e1715_ * _e1729_))));
                                let _e1748_ = (_e1732_ + (_e1732_ * (1f - (_e1720_ * _e1732_))));
                                let _e1758_ = (_e1742_ + (_e1742_ * (1f - (_e1705_ * _e1742_))));
                                let _e1760_ = (_e1744_ + (_e1744_ * (1f - (_e1710_ * _e1744_))));
                                let _e1762_ = (_e1746_ + (_e1746_ * (1f - (_e1715_ * _e1746_))));
                                let _e1764_ = (_e1748_ + (_e1748_ * (1f - (_e1720_ * _e1748_))));
                                let _e1774_ = (_e1758_ + (_e1758_ * (1f - (_e1705_ * _e1758_))));
                                let _e1776_ = (_e1760_ + (_e1760_ * (1f - (_e1710_ * _e1760_))));
                                let _e1778_ = (_e1762_ + (_e1762_ * (1f - (_e1715_ * _e1762_))));
                                let _e1780_ = (_e1764_ + (_e1764_ * (1f - (_e1720_ * _e1764_))));
                                let _e1781_ = (_e1693_ + _e1577_);
                                let _e1782_ = (_e1694_ + _e1582_);
                                let _e1783_ = (_e1695_ + _e1587_);
                                let _e1784_ = (_e1696_ + _e1592_);
                                if (_e1781_ != _e1781_) {
                                    phi_8858_ = true;
                                } else {
                                    phi_8858_ = (0.000001f >= _e1781_);
                                }
                                let _e1788_ = phi_8858_;
                                let _e1789_ = select(_e1781_, 0.000001f, _e1788_);
                                if (_e1782_ != _e1782_) {
                                    phi_8873_ = true;
                                } else {
                                    phi_8873_ = (0.000001f >= _e1782_);
                                }
                                let _e1793_ = phi_8873_;
                                let _e1794_ = select(_e1782_, 0.000001f, _e1793_);
                                if (_e1783_ != _e1783_) {
                                    phi_8888_ = true;
                                } else {
                                    phi_8888_ = (0.000001f >= _e1783_);
                                }
                                let _e1798_ = phi_8888_;
                                let _e1799_ = select(_e1783_, 0.000001f, _e1798_);
                                if (_e1784_ != _e1784_) {
                                    phi_8903_ = true;
                                } else {
                                    phi_8903_ = (0.000001f >= _e1784_);
                                }
                                let _e1803_ = phi_8903_;
                                let _e1804_ = select(_e1784_, 0.000001f, _e1803_);
                                let _e1807_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1789_)));
                                let _e1810_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1794_)));
                                let _e1813_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1799_)));
                                let _e1816_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1804_)));
                                let _e1826_ = (_e1807_ + (_e1807_ * (1f - (_e1789_ * _e1807_))));
                                let _e1828_ = (_e1810_ + (_e1810_ * (1f - (_e1794_ * _e1810_))));
                                let _e1830_ = (_e1813_ + (_e1813_ * (1f - (_e1799_ * _e1813_))));
                                let _e1832_ = (_e1816_ + (_e1816_ * (1f - (_e1804_ * _e1816_))));
                                let _e1842_ = (_e1826_ + (_e1826_ * (1f - (_e1789_ * _e1826_))));
                                let _e1844_ = (_e1828_ + (_e1828_ * (1f - (_e1794_ * _e1828_))));
                                let _e1846_ = (_e1830_ + (_e1830_ * (1f - (_e1799_ * _e1830_))));
                                let _e1848_ = (_e1832_ + (_e1832_ * (1f - (_e1804_ * _e1832_))));
                                let _e1885_ = (_e1563_ <= 1000000000000000f);
                                let _e1887_ = (_e1566_ <= 1000000000000000f);
                                let _e1889_ = (_e1569_ <= 1000000000000000f);
                                let _e1891_ = (_e1572_ <= 1000000000000000f);
                                let _e1922_ = select(select(select(_e1563_, (0.5f * (_e1577_ + (_e1593_ * _e1774_))), _e1885_), select(-0.0005f, -(((0.0005f * (_e1577_ * (_e1842_ + (_e1842_ * (1f - (_e1789_ * _e1842_)))))) * (1f + (_e1577_ * _e1774_)))), _e1885_), (_e1498_ <= 0f)), 0f, (_e1563_ <= 0f));
                                let _e1924_ = select(select(select(_e1566_, (0.5f * (_e1582_ + (_e1594_ * _e1776_))), _e1887_), select(-0.0005f, -(((0.0005f * (_e1582_ * (_e1844_ + (_e1844_ * (1f - (_e1794_ * _e1844_)))))) * (1f + (_e1582_ * _e1776_)))), _e1887_), (_e1500_ <= 0f)), 0f, (_e1566_ <= 0f));
                                let _e1926_ = select(select(select(_e1569_, (0.5f * (_e1587_ + (_e1595_ * _e1778_))), _e1889_), select(-0.0005f, -(((0.0005f * (_e1587_ * (_e1846_ + (_e1846_ * (1f - (_e1799_ * _e1846_)))))) * (1f + (_e1587_ * _e1778_)))), _e1889_), (_e1502_ <= 0f)), 0f, (_e1569_ <= 0f));
                                let _e1928_ = select(select(select(_e1572_, (0.5f * (_e1592_ + (_e1596_ * _e1780_))), _e1891_), select(-0.0005f, -(((0.0005f * (_e1592_ * (_e1848_ + (_e1848_ * (1f - (_e1804_ * _e1848_)))))) * (1f + (_e1592_ * _e1780_)))), _e1891_), (_e1504_ <= 0f)), 0f, (_e1572_ <= 0f));
                                let _e1941_ = (_e564_ <= 0f);
                                let _e1966_ = bitcast<u32>((((1f - (_e564_ * _e1467_)) * _e1485_) + (_e1467_ * select(((_e1495_ * _e1922_) + (_e564_ * (_e1529_ * (1f - (0.33333334f * (_e1529_ * _e1529_)))))), _e1922_, _e1941_))));
                                let _e1967_ = (_e1966_ & 2139095040u);
                                if (_e1967_ == 2139095040u) {
                                    phi_4947_ = true;
                                } else {
                                    phi_4947_ = (_e1967_ == 0u);
                                }
                                let _e1971_ = phi_4947_;
                                let _e1973_ = bitcast<f32>(select(_e1966_, 0u, _e1971_));
                                let _e1974_ = bitcast<u32>((((1f - (_e564_ * _e1472_)) * _e1488_) + (_e1472_ * select(((_e1495_ * _e1924_) + (_e564_ * (_e1534_ * (1f - (0.33333334f * (_e1534_ * _e1534_)))))), _e1924_, _e1941_))));
                                let _e1975_ = (_e1974_ & 2139095040u);
                                if (_e1975_ == 2139095040u) {
                                    phi_4963_ = true;
                                } else {
                                    phi_4963_ = (_e1975_ == 0u);
                                }
                                let _e1979_ = phi_4963_;
                                let _e1982_ = bitcast<u32>((((1f - (_e564_ * _e1477_)) * _e1491_) + (_e1477_ * select(((_e1495_ * _e1926_) + (_e564_ * (_e1539_ * (1f - (0.33333334f * (_e1539_ * _e1539_)))))), _e1926_, _e1941_))));
                                let _e1983_ = (_e1982_ & 2139095040u);
                                if (_e1983_ == 2139095040u) {
                                    phi_4979_ = true;
                                } else {
                                    phi_4979_ = (_e1983_ == 0u);
                                }
                                let _e1987_ = phi_4979_;
                                let _e1990_ = bitcast<u32>((((1f - (_e564_ * _e1482_)) * _e1494_) + (_e1482_ * select(((_e1495_ * _e1928_) + (_e564_ * (_e1544_ * (1f - (0.33333334f * (_e1544_ * _e1544_)))))), _e1928_, _e1941_))));
                                let _e1991_ = (_e1990_ & 2139095040u);
                                if (_e1991_ == 2139095040u) {
                                    phi_4995_ = true;
                                } else {
                                    phi_4995_ = (_e1991_ == 0u);
                                }
                                let _e1995_ = phi_4995_;
                                let _e2000_ = bitcast<u32>((_e625_ * _e1389_));
                                let _e2001_ = (_e2000_ & 2139095040u);
                                let _e2002_ = (_e2001_ == 2139095040u);
                                if _e2002_ {
                                    phi_5017_ = true;
                                } else {
                                    phi_5017_ = (_e2001_ == 0u);
                                }
                                let _e2005_ = phi_5017_;
                                if _e2002_ {
                                    phi_5033_ = true;
                                } else {
                                    phi_5033_ = (_e2001_ == 0u);
                                }
                                let _e2010_ = phi_5033_;
                                if _e2002_ {
                                    phi_5049_ = true;
                                } else {
                                    phi_5049_ = (_e2001_ == 0u);
                                }
                                let _e2015_ = phi_5049_;
                                if _e2002_ {
                                    phi_5065_ = true;
                                } else {
                                    phi_5065_ = (_e2001_ == 0u);
                                }
                                let _e2020_ = phi_5065_;
                                let _e2043_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e2000_, 0u, _e2005_))) & 2147483647u))), 0f, _e1433_));
                                let _e2044_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e2000_, 0u, _e2010_))) & 2147483647u))), 0f, _e1433_));
                                let _e2045_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e2000_, 0u, _e2015_))) & 2147483647u))), 0f, _e1433_));
                                let _e2046_ = (_e1438_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e2000_, 0u, _e2020_))) & 2147483647u))), 0f, _e1433_));
                                if (_e2043_ != _e2043_) {
                                    phi_9038_ = true;
                                } else {
                                    phi_9038_ = (0f >= _e2043_);
                                }
                                let _e2050_ = phi_9038_;
                                let _e2051_ = select(_e2043_, 0f, _e2050_);
                                if (_e2044_ != _e2044_) {
                                    phi_9053_ = true;
                                } else {
                                    phi_9053_ = (0f >= _e2044_);
                                }
                                let _e2055_ = phi_9053_;
                                let _e2056_ = select(_e2044_, 0f, _e2055_);
                                if (_e2045_ != _e2045_) {
                                    phi_9068_ = true;
                                } else {
                                    phi_9068_ = (0f >= _e2045_);
                                }
                                let _e2060_ = phi_9068_;
                                let _e2061_ = select(_e2045_, 0f, _e2060_);
                                if (_e2046_ != _e2046_) {
                                    phi_9083_ = true;
                                } else {
                                    phi_9083_ = (0f >= _e2046_);
                                }
                                let _e2065_ = phi_9083_;
                                let _e2066_ = select(_e2046_, 0f, _e2065_);
                                if (_e2051_ != _e2051_) {
                                    phi_9098_ = true;
                                } else {
                                    phi_9098_ = (1f <= _e2051_);
                                }
                                let _e2070_ = phi_9098_;
                                let _e2071_ = select(_e2051_, 1f, _e2070_);
                                if (_e2056_ != _e2056_) {
                                    phi_9113_ = true;
                                } else {
                                    phi_9113_ = (1f <= _e2056_);
                                }
                                let _e2075_ = phi_9113_;
                                let _e2076_ = select(_e2056_, 1f, _e2075_);
                                if (_e2061_ != _e2061_) {
                                    phi_9128_ = true;
                                } else {
                                    phi_9128_ = (1f <= _e2061_);
                                }
                                let _e2080_ = phi_9128_;
                                let _e2081_ = select(_e2061_, 1f, _e2080_);
                                if (_e2066_ != _e2066_) {
                                    phi_9143_ = true;
                                } else {
                                    phi_9143_ = (1f <= _e2066_);
                                }
                                let _e2085_ = phi_9143_;
                                let _e2086_ = select(_e2066_, 1f, _e2085_);
                                let _e2089_ = (_e540_ * _e1353_.member[0]);
                                let _e2092_ = (_e540_ * _e1353_.member[1]);
                                let _e2095_ = (_e540_ * _e1353_.member[2]);
                                let _e2098_ = (_e540_ * _e1353_.member[3]);
                                let _e2100_ = (_e1391_ - (_e1496_ * _e2089_));
                                let _e2102_ = (_e1391_ - (_e1496_ * _e2092_));
                                let _e2104_ = (_e1391_ - (_e1496_ * _e2095_));
                                let _e2106_ = (_e1391_ - (_e1496_ * _e2098_));
                                if (_e2100_ != _e2100_) {
                                    phi_9158_ = true;
                                } else {
                                    phi_9158_ = (-1f >= _e2100_);
                                }
                                let _e2110_ = phi_9158_;
                                let _e2111_ = select(_e2100_, -1f, _e2110_);
                                if (_e2102_ != _e2102_) {
                                    phi_9173_ = true;
                                } else {
                                    phi_9173_ = (-1f >= _e2102_);
                                }
                                let _e2115_ = phi_9173_;
                                let _e2116_ = select(_e2102_, -1f, _e2115_);
                                if (_e2104_ != _e2104_) {
                                    phi_9188_ = true;
                                } else {
                                    phi_9188_ = (-1f >= _e2104_);
                                }
                                let _e2120_ = phi_9188_;
                                let _e2121_ = select(_e2104_, -1f, _e2120_);
                                if (_e2106_ != _e2106_) {
                                    phi_9203_ = true;
                                } else {
                                    phi_9203_ = (-1f >= _e2106_);
                                }
                                let _e2125_ = phi_9203_;
                                let _e2126_ = select(_e2106_, -1f, _e2125_);
                                if (_e2111_ != _e2111_) {
                                    phi_9218_ = true;
                                } else {
                                    phi_9218_ = (1f <= _e2111_);
                                }
                                let _e2130_ = phi_9218_;
                                let _e2131_ = select(_e2111_, 1f, _e2130_);
                                if (_e2116_ != _e2116_) {
                                    phi_9233_ = true;
                                } else {
                                    phi_9233_ = (1f <= _e2116_);
                                }
                                let _e2135_ = phi_9233_;
                                let _e2136_ = select(_e2116_, 1f, _e2135_);
                                if (_e2121_ != _e2121_) {
                                    phi_9248_ = true;
                                } else {
                                    phi_9248_ = (1f <= _e2121_);
                                }
                                let _e2140_ = phi_9248_;
                                let _e2141_ = select(_e2121_, 1f, _e2140_);
                                if (_e2126_ != _e2126_) {
                                    phi_9263_ = true;
                                } else {
                                    phi_9263_ = (1f <= _e2126_);
                                }
                                let _e2145_ = phi_9263_;
                                let _e2146_ = select(_e2126_, 1f, _e2145_);
                                let _e2165_ = bitcast<f32>((bitcast<u32>(_e2100_) & 2147483647u));
                                let _e2168_ = bitcast<f32>((bitcast<u32>(_e2102_) & 2147483647u));
                                let _e2171_ = bitcast<f32>((bitcast<u32>(_e2104_) & 2147483647u));
                                let _e2174_ = bitcast<f32>((bitcast<u32>(_e2106_) & 2147483647u));
                                if (_e2165_ != _e2165_) {
                                    phi_9278_ = true;
                                } else {
                                    phi_9278_ = (1000000000000000f <= _e2165_);
                                }
                                let _e2178_ = phi_9278_;
                                let _e2179_ = select(_e2165_, 1000000000000000f, _e2178_);
                                if (_e2168_ != _e2168_) {
                                    phi_9293_ = true;
                                } else {
                                    phi_9293_ = (1000000000000000f <= _e2168_);
                                }
                                let _e2183_ = phi_9293_;
                                let _e2184_ = select(_e2168_, 1000000000000000f, _e2183_);
                                if (_e2171_ != _e2171_) {
                                    phi_9308_ = true;
                                } else {
                                    phi_9308_ = (1000000000000000f <= _e2171_);
                                }
                                let _e2188_ = phi_9308_;
                                let _e2189_ = select(_e2171_, 1000000000000000f, _e2188_);
                                if (_e2174_ != _e2174_) {
                                    phi_9323_ = true;
                                } else {
                                    phi_9323_ = (1000000000000000f <= _e2174_);
                                }
                                let _e2193_ = phi_9323_;
                                let _e2194_ = select(_e2174_, 1000000000000000f, _e2193_);
                                let _e2195_ = (_e2179_ * _e2179_);
                                let _e2196_ = (_e2184_ * _e2184_);
                                let _e2197_ = (_e2189_ * _e2189_);
                                let _e2198_ = (_e2194_ * _e2194_);
                                let _e2199_ = (_e2195_ + 0.0000010000001f);
                                let _e2200_ = (_e2196_ + 0.0000010000001f);
                                let _e2201_ = (_e2197_ + 0.0000010000001f);
                                let _e2202_ = (_e2198_ + 0.0000010000001f);
                                if (_e2199_ != _e2199_) {
                                    phi_9338_ = true;
                                } else {
                                    phi_9338_ = (0.000000000001f >= _e2199_);
                                }
                                let _e2206_ = phi_9338_;
                                let _e2207_ = select(_e2199_, 0.000000000001f, _e2206_);
                                if (_e2200_ != _e2200_) {
                                    phi_9353_ = true;
                                } else {
                                    phi_9353_ = (0.000000000001f >= _e2200_);
                                }
                                let _e2211_ = phi_9353_;
                                let _e2212_ = select(_e2200_, 0.000000000001f, _e2211_);
                                if (_e2201_ != _e2201_) {
                                    phi_9368_ = true;
                                } else {
                                    phi_9368_ = (0.000000000001f >= _e2201_);
                                }
                                let _e2216_ = phi_9368_;
                                let _e2217_ = select(_e2201_, 0.000000000001f, _e2216_);
                                if (_e2202_ != _e2202_) {
                                    phi_9383_ = true;
                                } else {
                                    phi_9383_ = (0.000000000001f >= _e2202_);
                                }
                                let _e2221_ = phi_9383_;
                                let _e2222_ = select(_e2202_, 0.000000000001f, _e2221_);
                                let _e2223_ = (_e2207_ * 0.5f);
                                let _e2224_ = (_e2212_ * 0.5f);
                                let _e2225_ = (_e2217_ * 0.5f);
                                let _e2226_ = (_e2222_ * 0.5f);
                                let _e2231_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2207_) >> bitcast<u32>(1i))));
                                let _e2236_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2212_) >> bitcast<u32>(1i))));
                                let _e2241_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2217_) >> bitcast<u32>(1i))));
                                let _e2246_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2222_) >> bitcast<u32>(1i))));
                                let _e2259_ = (_e2231_ * (1.5f - (_e2223_ * (_e2231_ * _e2231_))));
                                let _e2260_ = (_e2236_ * (1.5f - (_e2224_ * (_e2236_ * _e2236_))));
                                let _e2261_ = (_e2241_ * (1.5f - (_e2225_ * (_e2241_ * _e2241_))));
                                let _e2262_ = (_e2246_ * (1.5f - (_e2226_ * (_e2246_ * _e2246_))));
                                let _e2275_ = (_e2259_ * (1.5f - (_e2223_ * (_e2259_ * _e2259_))));
                                let _e2276_ = (_e2260_ * (1.5f - (_e2224_ * (_e2260_ * _e2260_))));
                                let _e2277_ = (_e2261_ * (1.5f - (_e2225_ * (_e2261_ * _e2261_))));
                                let _e2278_ = (_e2262_ * (1.5f - (_e2226_ * (_e2262_ * _e2262_))));
                                let _e2295_ = (_e2199_ * (_e2275_ * (1.5f - (_e2223_ * (_e2275_ * _e2275_)))));
                                let _e2296_ = (_e2200_ * (_e2276_ * (1.5f - (_e2224_ * (_e2276_ * _e2276_)))));
                                let _e2297_ = (_e2201_ * (_e2277_ * (1.5f - (_e2225_ * (_e2277_ * _e2277_)))));
                                let _e2298_ = (_e2202_ * (_e2278_ * (1.5f - (_e2226_ * (_e2278_ * _e2278_)))));
                                let _e2299_ = (_e2295_ + 0.001f);
                                let _e2300_ = (_e2296_ + 0.001f);
                                let _e2301_ = (_e2297_ + 0.001f);
                                let _e2302_ = (_e2298_ + 0.001f);
                                if (_e2299_ != _e2299_) {
                                    phi_9398_ = true;
                                } else {
                                    phi_9398_ = (0.000001f >= _e2299_);
                                }
                                let _e2306_ = phi_9398_;
                                let _e2307_ = select(_e2299_, 0.000001f, _e2306_);
                                if (_e2300_ != _e2300_) {
                                    phi_9413_ = true;
                                } else {
                                    phi_9413_ = (0.000001f >= _e2300_);
                                }
                                let _e2311_ = phi_9413_;
                                let _e2312_ = select(_e2300_, 0.000001f, _e2311_);
                                if (_e2301_ != _e2301_) {
                                    phi_9428_ = true;
                                } else {
                                    phi_9428_ = (0.000001f >= _e2301_);
                                }
                                let _e2316_ = phi_9428_;
                                let _e2317_ = select(_e2301_, 0.000001f, _e2316_);
                                if (_e2302_ != _e2302_) {
                                    phi_9443_ = true;
                                } else {
                                    phi_9443_ = (0.000001f >= _e2302_);
                                }
                                let _e2321_ = phi_9443_;
                                let _e2322_ = select(_e2302_, 0.000001f, _e2321_);
                                let _e2325_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2307_)));
                                let _e2328_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2312_)));
                                let _e2331_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2317_)));
                                let _e2334_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2322_)));
                                let _e2344_ = (_e2325_ + (_e2325_ * (1f - (_e2307_ * _e2325_))));
                                let _e2346_ = (_e2328_ + (_e2328_ * (1f - (_e2312_ * _e2328_))));
                                let _e2348_ = (_e2331_ + (_e2331_ * (1f - (_e2317_ * _e2331_))));
                                let _e2350_ = (_e2334_ + (_e2334_ * (1f - (_e2322_ * _e2334_))));
                                let _e2360_ = (_e2344_ + (_e2344_ * (1f - (_e2307_ * _e2344_))));
                                let _e2362_ = (_e2346_ + (_e2346_ * (1f - (_e2312_ * _e2346_))));
                                let _e2364_ = (_e2348_ + (_e2348_ * (1f - (_e2317_ * _e2348_))));
                                let _e2366_ = (_e2350_ + (_e2350_ * (1f - (_e2322_ * _e2350_))));
                                let _e2376_ = (_e2360_ + (_e2360_ * (1f - (_e2307_ * _e2360_))));
                                let _e2378_ = (_e2362_ + (_e2362_ * (1f - (_e2312_ * _e2362_))));
                                let _e2380_ = (_e2364_ + (_e2364_ * (1f - (_e2317_ * _e2364_))));
                                let _e2382_ = (_e2366_ + (_e2366_ * (1f - (_e2322_ * _e2366_))));
                                let _e2383_ = (_e2295_ + _e2179_);
                                let _e2384_ = (_e2296_ + _e2184_);
                                let _e2385_ = (_e2297_ + _e2189_);
                                let _e2386_ = (_e2298_ + _e2194_);
                                if (_e2383_ != _e2383_) {
                                    phi_9458_ = true;
                                } else {
                                    phi_9458_ = (0.000001f >= _e2383_);
                                }
                                let _e2390_ = phi_9458_;
                                let _e2391_ = select(_e2383_, 0.000001f, _e2390_);
                                if (_e2384_ != _e2384_) {
                                    phi_9473_ = true;
                                } else {
                                    phi_9473_ = (0.000001f >= _e2384_);
                                }
                                let _e2395_ = phi_9473_;
                                let _e2396_ = select(_e2384_, 0.000001f, _e2395_);
                                if (_e2385_ != _e2385_) {
                                    phi_9488_ = true;
                                } else {
                                    phi_9488_ = (0.000001f >= _e2385_);
                                }
                                let _e2400_ = phi_9488_;
                                let _e2401_ = select(_e2385_, 0.000001f, _e2400_);
                                if (_e2386_ != _e2386_) {
                                    phi_9503_ = true;
                                } else {
                                    phi_9503_ = (0.000001f >= _e2386_);
                                }
                                let _e2405_ = phi_9503_;
                                let _e2406_ = select(_e2386_, 0.000001f, _e2405_);
                                let _e2409_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2391_)));
                                let _e2412_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2396_)));
                                let _e2415_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2401_)));
                                let _e2418_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2406_)));
                                let _e2428_ = (_e2409_ + (_e2409_ * (1f - (_e2391_ * _e2409_))));
                                let _e2430_ = (_e2412_ + (_e2412_ * (1f - (_e2396_ * _e2412_))));
                                let _e2432_ = (_e2415_ + (_e2415_ * (1f - (_e2401_ * _e2415_))));
                                let _e2434_ = (_e2418_ + (_e2418_ * (1f - (_e2406_ * _e2418_))));
                                let _e2444_ = (_e2428_ + (_e2428_ * (1f - (_e2391_ * _e2428_))));
                                let _e2446_ = (_e2430_ + (_e2430_ * (1f - (_e2396_ * _e2430_))));
                                let _e2448_ = (_e2432_ + (_e2432_ * (1f - (_e2401_ * _e2432_))));
                                let _e2450_ = (_e2434_ + (_e2434_ * (1f - (_e2406_ * _e2434_))));
                                let _e2487_ = (_e2165_ <= 1000000000000000f);
                                let _e2489_ = (_e2168_ <= 1000000000000000f);
                                let _e2491_ = (_e2171_ <= 1000000000000000f);
                                let _e2493_ = (_e2174_ <= 1000000000000000f);
                                let _e2524_ = select(select(select(_e2165_, (0.5f * (_e2179_ + (_e2195_ * _e2376_))), _e2487_), select(-0.0005f, -(((0.0005f * (_e2179_ * (_e2444_ + (_e2444_ * (1f - (_e2391_ * _e2444_)))))) * (1f + (_e2179_ * _e2376_)))), _e2487_), (_e2100_ <= 0f)), 0f, (_e2165_ <= 0f));
                                let _e2526_ = select(select(select(_e2168_, (0.5f * (_e2184_ + (_e2196_ * _e2378_))), _e2489_), select(-0.0005f, -(((0.0005f * (_e2184_ * (_e2446_ + (_e2446_ * (1f - (_e2396_ * _e2446_)))))) * (1f + (_e2184_ * _e2378_)))), _e2489_), (_e2102_ <= 0f)), 0f, (_e2168_ <= 0f));
                                let _e2528_ = select(select(select(_e2171_, (0.5f * (_e2189_ + (_e2197_ * _e2380_))), _e2491_), select(-0.0005f, -(((0.0005f * (_e2189_ * (_e2448_ + (_e2448_ * (1f - (_e2401_ * _e2448_)))))) * (1f + (_e2189_ * _e2380_)))), _e2491_), (_e2104_ <= 0f)), 0f, (_e2171_ <= 0f));
                                let _e2530_ = select(select(select(_e2174_, (0.5f * (_e2194_ + (_e2198_ * _e2382_))), _e2493_), select(-0.0005f, -(((0.0005f * (_e2194_ * (_e2450_ + (_e2450_ * (1f - (_e2406_ * _e2450_)))))) * (1f + (_e2194_ * _e2382_)))), _e2493_), (_e2106_ <= 0f)), 0f, (_e2174_ <= 0f));
                                let _e2567_ = bitcast<u32>((((1f - (_e564_ * _e2071_)) * _e2089_) + (_e2071_ * select(((_e1495_ * _e2524_) + (_e564_ * (_e2131_ * (1f - (0.33333334f * (_e2131_ * _e2131_)))))), _e2524_, _e1941_))));
                                let _e2568_ = (_e2567_ & 2139095040u);
                                if (_e2568_ == 2139095040u) {
                                    phi_5773_ = true;
                                } else {
                                    phi_5773_ = (_e2568_ == 0u);
                                }
                                let _e2572_ = phi_5773_;
                                let _e2574_ = bitcast<f32>(select(_e2567_, 0u, _e2572_));
                                let _e2575_ = bitcast<u32>((((1f - (_e564_ * _e2076_)) * _e2092_) + (_e2076_ * select(((_e1495_ * _e2526_) + (_e564_ * (_e2136_ * (1f - (0.33333334f * (_e2136_ * _e2136_)))))), _e2526_, _e1941_))));
                                let _e2576_ = (_e2575_ & 2139095040u);
                                if (_e2576_ == 2139095040u) {
                                    phi_5789_ = true;
                                } else {
                                    phi_5789_ = (_e2576_ == 0u);
                                }
                                let _e2580_ = phi_5789_;
                                let _e2583_ = bitcast<u32>((((1f - (_e564_ * _e2081_)) * _e2095_) + (_e2081_ * select(((_e1495_ * _e2528_) + (_e564_ * (_e2141_ * (1f - (0.33333334f * (_e2141_ * _e2141_)))))), _e2528_, _e1941_))));
                                let _e2584_ = (_e2583_ & 2139095040u);
                                if (_e2584_ == 2139095040u) {
                                    phi_5805_ = true;
                                } else {
                                    phi_5805_ = (_e2584_ == 0u);
                                }
                                let _e2588_ = phi_5805_;
                                let _e2591_ = bitcast<u32>((((1f - (_e564_ * _e2086_)) * _e2098_) + (_e2086_ * select(((_e1495_ * _e2530_) + (_e564_ * (_e2146_ * (1f - (0.33333334f * (_e2146_ * _e2146_)))))), _e2530_, _e1941_))));
                                let _e2592_ = (_e2591_ & 2139095040u);
                                if (_e2592_ == 2139095040u) {
                                    phi_5821_ = true;
                                } else {
                                    phi_5821_ = (_e2592_ == 0u);
                                }
                                let _e2596_ = phi_5821_;
                                let _e2602_ = (0.5f * (_e1973_ + _e2574_));
                                let _e2609_ = (1024u + _e1370_.member_1_);
                                if (_e2609_ < 1536u) {
                                } else {
                                    phi_12169_ = true;
                                    break;
                                }
                                scan_scratch[_e2609_] = (_e1973_ + (_e570_ * (_e2602_ - _e1973_)));
                                let _e2612_ = (1280u + _e1370_.member_1_);
                                if (_e2612_ < 1536u) {
                                } else {
                                    phi_12169_ = true;
                                    break;
                                }
                                scan_scratch[_e2612_] = (_e2574_ + (_e570_ * (_e2602_ - _e2574_)));
                                phi_4114_ = type_22_(array<f32, 4>(_e2574_, bitcast<f32>(select(_e2575_, 0u, _e2580_)), bitcast<f32>(select(_e2583_, 0u, _e2588_)), bitcast<f32>(select(_e2591_, 0u, _e2596_))));
                                phi_4116_ = type_22_(array<f32, 4>(_e1973_, bitcast<f32>(select(_e1974_, 0u, _e1979_)), bitcast<f32>(select(_e1982_, 0u, _e1987_)), bitcast<f32>(select(_e1990_, 0u, _e1995_))));
                            } else {
                                phi_4114_ = type_22_();
                                phi_4116_ = type_22_();
                            }
                            let _e2616_ = phi_4114_;
                            let _e2618_ = phi_4116_;
                            phi_4110_ = _e1368_;
                            phi_4113_ = _e2616_;
                            phi_4115_ = _e2618_;
                            phi_12169_ = _e1306_;
                            if !(_e1374_) {
                                break;
                            }
                        }
                        let _e2621_ = phi_12169_;
                        phi_12454_ = _e2621_;
                        if _e2621_ {
                            break;
                        }
                        phi_12851_ = _e2621_;
                    } else {
                        phi_12851_ = _e1306_;
                    }
                    let _e2623_ = phi_12851_;
                    workgroupBarrier();
                    if _e160_ {
                        if _e885_ {
                        } else {
                            phi_12454_ = true;
                            break;
                        }
                        let _e2624_ = scan_scratch[_e884_];
                        phi_5909_ = _e2624_;
                    } else {
                        phi_5909_ = 0f;
                    }
                    let _e2626_ = phi_5909_;
                    if _e160_ {
                        if _e889_ {
                        } else {
                            phi_12454_ = true;
                            break;
                        }
                        let _e2627_ = scan_scratch[_e888_];
                        phi_5921_ = _e2627_;
                    } else {
                        phi_5921_ = 0f;
                    }
                    let _e2629_ = phi_5921_;
                    let _e2631_ = (1f - (_e775_ * _e2626_));
                    if (_e2631_ != _e2631_) {
                        phi_9638_ = true;
                    } else {
                        phi_9638_ = (0.05f >= _e2631_);
                    }
                    let _e2635_ = phi_9638_;
                    let _e2638_ = (1f - (_e775_ * _e2629_));
                    if (_e2638_ != _e2638_) {
                        phi_9653_ = true;
                    } else {
                        phi_9653_ = (0.05f >= _e2638_);
                    }
                    let _e2642_ = phi_9653_;
                    if _e160_ {
                        let _e2648_ = bitcast<u32>((((_e1108_.member * _e814_) * select(_e2631_, 0.05f, _e2635_)) + (_e853_ * _e2626_)));
                        let _e2649_ = (_e2648_ & 2139095040u);
                        if (_e2649_ == 2139095040u) {
                            phi_5945_ = true;
                        } else {
                            phi_5945_ = (_e2649_ == 0u);
                        }
                        let _e2653_ = phi_5945_;
                        phi_5951_ = bitcast<f32>(select(_e2648_, 0u, _e2653_));
                    } else {
                        phi_5951_ = 0f;
                    }
                    let _e2657_ = phi_5951_;
                    if _e160_ {
                        let _e2662_ = bitcast<u32>((((_e1345_.member * _e814_) * select(_e2638_, 0.05f, _e2642_)) + (_e853_ * _e2629_)));
                        let _e2663_ = (_e2662_ & 2139095040u);
                        if (_e2663_ == 2139095040u) {
                            phi_5969_ = true;
                        } else {
                            phi_5969_ = (_e2663_ == 0u);
                        }
                        let _e2667_ = phi_5969_;
                        phi_5975_ = bitcast<f32>(select(_e2662_, 0u, _e2667_));
                    } else {
                        phi_5975_ = 0f;
                    }
                    let _e2671_ = phi_5975_;
                    workgroupBarrier();
                    if _e869_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e2657_;
                    if _e873_ {
                    } else {
                        phi_12454_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e2671_;
                    workgroupBarrier();
                    let _e2672_ = (_e114_ > 0u);
                    if _e2672_ {
                        if _e160_ {
                            let _e2673_ = (_e114_ - 1u);
                            if (_e2673_ < 1536u) {
                            } else {
                                phi_12454_ = true;
                                break;
                            }
                            let _e2676_ = scan_scratch[_e2673_];
                            phi_6004_ = _e2676_;
                        } else {
                            phi_6004_ = f32();
                        }
                        let _e2678_ = phi_6004_;
                        phi_6006_ = _e2678_;
                        phi_6007_ = select(true, false, _e160_);
                    } else {
                        phi_6006_ = f32();
                        phi_6007_ = true;
                    }
                    let _e2681_ = phi_6006_;
                    let _e2683_ = phi_6007_;
                    let _e2684_ = select(_e2681_, 0f, _e2683_);
                    if _e2672_ {
                        if _e160_ {
                            let _e2685_ = (_e114_ + 255u);
                            if (_e2685_ < 1536u) {
                            } else {
                                phi_12454_ = true;
                                break;
                            }
                            let _e2688_ = scan_scratch[_e2685_];
                            phi_6028_ = _e2688_;
                        } else {
                            phi_6028_ = f32();
                        }
                        let _e2690_ = phi_6028_;
                        phi_6030_ = _e2690_;
                        phi_6031_ = select(true, false, _e160_);
                    } else {
                        phi_6030_ = f32();
                        phi_6031_ = true;
                    }
                    let _e2693_ = phi_6030_;
                    let _e2695_ = phi_6031_;
                    let _e2696_ = select(_e2693_, 0f, _e2695_);
                    let _e2701_ = select(0f, _e2684_, (bitcast<f32>((bitcast<u32>(_e2684_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2702_ = (_e2701_ != _e2701_);
                    if _e2702_ {
                        phi_9668_ = true;
                    } else {
                        phi_9668_ = (-1000000000000000000f >= _e2701_);
                    }
                    let _e2705_ = phi_9668_;
                    let _e2706_ = select(_e2701_, -1000000000000000000f, _e2705_);
                    if _e2702_ {
                        phi_9683_ = true;
                    } else {
                        phi_9683_ = (-1000000000000000000f >= _e2701_);
                    }
                    let _e2709_ = phi_9683_;
                    let _e2710_ = select(_e2701_, -1000000000000000000f, _e2709_);
                    if _e2702_ {
                        phi_9698_ = true;
                    } else {
                        phi_9698_ = (-1000000000000000000f >= _e2701_);
                    }
                    let _e2713_ = phi_9698_;
                    let _e2714_ = select(_e2701_, -1000000000000000000f, _e2713_);
                    if _e2702_ {
                        phi_9713_ = true;
                    } else {
                        phi_9713_ = (-1000000000000000000f >= _e2701_);
                    }
                    let _e2717_ = phi_9713_;
                    let _e2718_ = select(_e2701_, -1000000000000000000f, _e2717_);
                    if (_e2706_ != _e2706_) {
                        phi_9728_ = true;
                    } else {
                        phi_9728_ = (1000000000000000000f <= _e2706_);
                    }
                    let _e2722_ = phi_9728_;
                    let _e2723_ = select(_e2706_, 1000000000000000000f, _e2722_);
                    if (_e2710_ != _e2710_) {
                        phi_9743_ = true;
                    } else {
                        phi_9743_ = (1000000000000000000f <= _e2710_);
                    }
                    let _e2727_ = phi_9743_;
                    let _e2728_ = select(_e2710_, 1000000000000000000f, _e2727_);
                    if (_e2714_ != _e2714_) {
                        phi_9758_ = true;
                    } else {
                        phi_9758_ = (1000000000000000000f <= _e2714_);
                    }
                    let _e2732_ = phi_9758_;
                    let _e2733_ = select(_e2714_, 1000000000000000000f, _e2732_);
                    if (_e2718_ != _e2718_) {
                        phi_9773_ = true;
                    } else {
                        phi_9773_ = (1000000000000000000f <= _e2718_);
                    }
                    let _e2737_ = phi_9773_;
                    let _e2738_ = select(_e2718_, 1000000000000000000f, _e2737_);
                    let _e2743_ = select(0f, _e2657_, (bitcast<f32>((bitcast<u32>(_e2657_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2744_ = (_e2743_ != _e2743_);
                    if _e2744_ {
                        phi_9788_ = true;
                    } else {
                        phi_9788_ = (-1000000000000000000f >= _e2743_);
                    }
                    let _e2747_ = phi_9788_;
                    let _e2748_ = select(_e2743_, -1000000000000000000f, _e2747_);
                    if _e2744_ {
                        phi_9803_ = true;
                    } else {
                        phi_9803_ = (-1000000000000000000f >= _e2743_);
                    }
                    let _e2751_ = phi_9803_;
                    let _e2752_ = select(_e2743_, -1000000000000000000f, _e2751_);
                    if _e2744_ {
                        phi_9818_ = true;
                    } else {
                        phi_9818_ = (-1000000000000000000f >= _e2743_);
                    }
                    let _e2755_ = phi_9818_;
                    let _e2756_ = select(_e2743_, -1000000000000000000f, _e2755_);
                    if _e2744_ {
                        phi_9833_ = true;
                    } else {
                        phi_9833_ = (-1000000000000000000f >= _e2743_);
                    }
                    let _e2759_ = phi_9833_;
                    let _e2760_ = select(_e2743_, -1000000000000000000f, _e2759_);
                    if (_e2748_ != _e2748_) {
                        phi_9848_ = true;
                    } else {
                        phi_9848_ = (1000000000000000000f <= _e2748_);
                    }
                    let _e2764_ = phi_9848_;
                    let _e2765_ = select(_e2748_, 1000000000000000000f, _e2764_);
                    if (_e2752_ != _e2752_) {
                        phi_9863_ = true;
                    } else {
                        phi_9863_ = (1000000000000000000f <= _e2752_);
                    }
                    let _e2769_ = phi_9863_;
                    let _e2770_ = select(_e2752_, 1000000000000000000f, _e2769_);
                    if (_e2756_ != _e2756_) {
                        phi_9878_ = true;
                    } else {
                        phi_9878_ = (1000000000000000000f <= _e2756_);
                    }
                    let _e2774_ = phi_9878_;
                    let _e2775_ = select(_e2756_, 1000000000000000000f, _e2774_);
                    if (_e2760_ != _e2760_) {
                        phi_9893_ = true;
                    } else {
                        phi_9893_ = (1000000000000000000f <= _e2760_);
                    }
                    let _e2779_ = phi_9893_;
                    let _e2780_ = select(_e2760_, 1000000000000000000f, _e2779_);
                    let _e2781_ = array<f32, 4>(_e2765_, _e2770_, _e2775_, _e2780_);
                    let _e2782_ = (_e2723_ != _e2723_);
                    if _e2782_ {
                        phi_9908_ = true;
                    } else {
                        phi_9908_ = (_e2765_ <= _e2723_);
                    }
                    let _e2785_ = phi_9908_;
                    let _e2786_ = select(_e2723_, _e2765_, _e2785_);
                    let _e2787_ = (_e2728_ != _e2728_);
                    if _e2787_ {
                        phi_9923_ = true;
                    } else {
                        phi_9923_ = (_e2770_ <= _e2728_);
                    }
                    let _e2790_ = phi_9923_;
                    let _e2791_ = select(_e2728_, _e2770_, _e2790_);
                    let _e2792_ = (_e2733_ != _e2733_);
                    if _e2792_ {
                        phi_9938_ = true;
                    } else {
                        phi_9938_ = (_e2775_ <= _e2733_);
                    }
                    let _e2795_ = phi_9938_;
                    let _e2796_ = select(_e2733_, _e2775_, _e2795_);
                    let _e2797_ = (_e2738_ != _e2738_);
                    if _e2797_ {
                        phi_9953_ = true;
                    } else {
                        phi_9953_ = (_e2780_ <= _e2738_);
                    }
                    let _e2800_ = phi_9953_;
                    let _e2801_ = select(_e2738_, _e2780_, _e2800_);
                    if _e2782_ {
                        phi_9968_ = true;
                    } else {
                        phi_9968_ = (_e2765_ >= _e2723_);
                    }
                    let _e2804_ = phi_9968_;
                    let _e2805_ = select(_e2723_, _e2765_, _e2804_);
                    if _e2787_ {
                        phi_9983_ = true;
                    } else {
                        phi_9983_ = (_e2770_ >= _e2728_);
                    }
                    let _e2808_ = phi_9983_;
                    let _e2809_ = select(_e2728_, _e2770_, _e2808_);
                    if _e2792_ {
                        phi_9998_ = true;
                    } else {
                        phi_9998_ = (_e2775_ >= _e2733_);
                    }
                    let _e2812_ = phi_9998_;
                    let _e2813_ = select(_e2733_, _e2775_, _e2812_);
                    if _e2797_ {
                        phi_10013_ = true;
                    } else {
                        phi_10013_ = (_e2780_ >= _e2738_);
                    }
                    let _e2816_ = phi_10013_;
                    let _e2817_ = select(_e2738_, _e2780_, _e2816_);
                    let _e2818_ = (_e2786_ != _e2786_);
                    if _e2818_ {
                        phi_10028_ = true;
                    } else {
                        phi_10028_ = (-1f >= _e2786_);
                    }
                    let _e2821_ = phi_10028_;
                    let _e2822_ = select(_e2786_, -1f, _e2821_);
                    let _e2823_ = (_e2791_ != _e2791_);
                    if _e2823_ {
                        phi_10043_ = true;
                    } else {
                        phi_10043_ = (-1f >= _e2791_);
                    }
                    let _e2826_ = phi_10043_;
                    let _e2827_ = select(_e2791_, -1f, _e2826_);
                    let _e2828_ = (_e2796_ != _e2796_);
                    if _e2828_ {
                        phi_10058_ = true;
                    } else {
                        phi_10058_ = (-1f >= _e2796_);
                    }
                    let _e2831_ = phi_10058_;
                    let _e2832_ = select(_e2796_, -1f, _e2831_);
                    let _e2833_ = (_e2801_ != _e2801_);
                    if _e2833_ {
                        phi_10073_ = true;
                    } else {
                        phi_10073_ = (-1f >= _e2801_);
                    }
                    let _e2836_ = phi_10073_;
                    let _e2837_ = select(_e2801_, -1f, _e2836_);
                    if (_e2822_ != _e2822_) {
                        phi_10088_ = true;
                    } else {
                        phi_10088_ = (1f <= _e2822_);
                    }
                    let _e2841_ = phi_10088_;
                    let _e2842_ = select(_e2822_, 1f, _e2841_);
                    if (_e2827_ != _e2827_) {
                        phi_10103_ = true;
                    } else {
                        phi_10103_ = (1f <= _e2827_);
                    }
                    let _e2846_ = phi_10103_;
                    let _e2847_ = select(_e2827_, 1f, _e2846_);
                    if (_e2832_ != _e2832_) {
                        phi_10118_ = true;
                    } else {
                        phi_10118_ = (1f <= _e2832_);
                    }
                    let _e2851_ = phi_10118_;
                    let _e2852_ = select(_e2832_, 1f, _e2851_);
                    if (_e2837_ != _e2837_) {
                        phi_10133_ = true;
                    } else {
                        phi_10133_ = (1f <= _e2837_);
                    }
                    let _e2856_ = phi_10133_;
                    let _e2857_ = select(_e2837_, 1f, _e2856_);
                    let _e2858_ = (_e2805_ != _e2805_);
                    if _e2858_ {
                        phi_10148_ = true;
                    } else {
                        phi_10148_ = (-1f >= _e2805_);
                    }
                    let _e2861_ = phi_10148_;
                    let _e2862_ = select(_e2805_, -1f, _e2861_);
                    let _e2863_ = (_e2809_ != _e2809_);
                    if _e2863_ {
                        phi_10163_ = true;
                    } else {
                        phi_10163_ = (-1f >= _e2809_);
                    }
                    let _e2866_ = phi_10163_;
                    let _e2867_ = select(_e2809_, -1f, _e2866_);
                    let _e2868_ = (_e2813_ != _e2813_);
                    if _e2868_ {
                        phi_10178_ = true;
                    } else {
                        phi_10178_ = (-1f >= _e2813_);
                    }
                    let _e2871_ = phi_10178_;
                    let _e2872_ = select(_e2813_, -1f, _e2871_);
                    let _e2873_ = (_e2817_ != _e2817_);
                    if _e2873_ {
                        phi_10193_ = true;
                    } else {
                        phi_10193_ = (-1f >= _e2817_);
                    }
                    let _e2876_ = phi_10193_;
                    let _e2877_ = select(_e2817_, -1f, _e2876_);
                    if (_e2862_ != _e2862_) {
                        phi_10208_ = true;
                    } else {
                        phi_10208_ = (1f <= _e2862_);
                    }
                    let _e2881_ = phi_10208_;
                    let _e2882_ = select(_e2862_, 1f, _e2881_);
                    if (_e2867_ != _e2867_) {
                        phi_10223_ = true;
                    } else {
                        phi_10223_ = (1f <= _e2867_);
                    }
                    let _e2886_ = phi_10223_;
                    let _e2887_ = select(_e2867_, 1f, _e2886_);
                    if (_e2872_ != _e2872_) {
                        phi_10238_ = true;
                    } else {
                        phi_10238_ = (1f <= _e2872_);
                    }
                    let _e2891_ = phi_10238_;
                    let _e2892_ = select(_e2872_, 1f, _e2891_);
                    if (_e2877_ != _e2877_) {
                        phi_10253_ = true;
                    } else {
                        phi_10253_ = (1f <= _e2877_);
                    }
                    let _e2896_ = phi_10253_;
                    let _e2897_ = select(_e2877_, 1f, _e2896_);
                    let _e2902_ = (0.5f * (_e2882_ + _e2842_));
                    let _e2903_ = (0.5f * (_e2887_ + _e2847_));
                    let _e2904_ = (0.5f * (_e2892_ + _e2852_));
                    let _e2905_ = (0.5f * (_e2897_ + _e2857_));
                    let _e2906_ = (_e2882_ - _e2842_);
                    let _e2907_ = (_e2887_ - _e2847_);
                    let _e2908_ = (_e2892_ - _e2852_);
                    let _e2909_ = (_e2897_ - _e2857_);
                    let _e2910_ = (0.5f * _e2906_);
                    let _e2911_ = (0.5f * _e2907_);
                    let _e2912_ = (0.5f * _e2908_);
                    let _e2913_ = (0.5f * _e2909_);
                    let _e2914_ = (_e2902_ * _e2902_);
                    let _e2915_ = (_e2903_ * _e2903_);
                    let _e2916_ = (_e2904_ * _e2904_);
                    let _e2917_ = (_e2905_ * _e2905_);
                    let _e2918_ = (_e2910_ * _e2910_);
                    let _e2919_ = (_e2911_ * _e2911_);
                    let _e2920_ = (_e2912_ * _e2912_);
                    let _e2921_ = (_e2913_ * _e2913_);
                    if (_e2906_ != _e2906_) {
                        phi_10268_ = true;
                    } else {
                        phi_10268_ = (0f >= _e2906_);
                    }
                    let _e2925_ = phi_10268_;
                    let _e2926_ = select(_e2906_, 0f, _e2925_);
                    if (_e2907_ != _e2907_) {
                        phi_10283_ = true;
                    } else {
                        phi_10283_ = (0f >= _e2907_);
                    }
                    let _e2930_ = phi_10283_;
                    let _e2931_ = select(_e2907_, 0f, _e2930_);
                    if (_e2908_ != _e2908_) {
                        phi_10298_ = true;
                    } else {
                        phi_10298_ = (0f >= _e2908_);
                    }
                    let _e2935_ = phi_10298_;
                    let _e2936_ = select(_e2908_, 0f, _e2935_);
                    if (_e2909_ != _e2909_) {
                        phi_10313_ = true;
                    } else {
                        phi_10313_ = (0f >= _e2909_);
                    }
                    let _e2940_ = phi_10313_;
                    let _e2941_ = select(_e2909_, 0f, _e2940_);
                    let _e2942_ = (_e2805_ - _e2786_);
                    let _e2943_ = (_e2809_ - _e2791_);
                    let _e2944_ = (_e2813_ - _e2796_);
                    let _e2945_ = (_e2817_ - _e2801_);
                    if _e2858_ {
                        phi_10328_ = true;
                    } else {
                        phi_10328_ = (-1f <= _e2805_);
                    }
                    let _e2948_ = phi_10328_;
                    if _e2863_ {
                        phi_10343_ = true;
                    } else {
                        phi_10343_ = (-1f <= _e2809_);
                    }
                    let _e2952_ = phi_10343_;
                    if _e2868_ {
                        phi_10358_ = true;
                    } else {
                        phi_10358_ = (-1f <= _e2813_);
                    }
                    let _e2956_ = phi_10358_;
                    if _e2873_ {
                        phi_10373_ = true;
                    } else {
                        phi_10373_ = (-1f <= _e2817_);
                    }
                    let _e2960_ = phi_10373_;
                    let _e2962_ = (select(_e2805_, -1f, _e2948_) - _e2786_);
                    let _e2963_ = (select(_e2809_, -1f, _e2952_) - _e2791_);
                    let _e2964_ = (select(_e2813_, -1f, _e2956_) - _e2796_);
                    let _e2965_ = (select(_e2817_, -1f, _e2960_) - _e2801_);
                    if (_e2962_ != _e2962_) {
                        phi_10388_ = true;
                    } else {
                        phi_10388_ = (0f >= _e2962_);
                    }
                    let _e2969_ = phi_10388_;
                    if (_e2963_ != _e2963_) {
                        phi_10403_ = true;
                    } else {
                        phi_10403_ = (0f >= _e2963_);
                    }
                    let _e2974_ = phi_10403_;
                    if (_e2964_ != _e2964_) {
                        phi_10418_ = true;
                    } else {
                        phi_10418_ = (0f >= _e2964_);
                    }
                    let _e2979_ = phi_10418_;
                    if (_e2965_ != _e2965_) {
                        phi_10433_ = true;
                    } else {
                        phi_10433_ = (0f >= _e2965_);
                    }
                    let _e2984_ = phi_10433_;
                    if _e2818_ {
                        phi_10448_ = true;
                    } else {
                        phi_10448_ = (1f >= _e2786_);
                    }
                    let _e2988_ = phi_10448_;
                    if _e2823_ {
                        phi_10463_ = true;
                    } else {
                        phi_10463_ = (1f >= _e2791_);
                    }
                    let _e2992_ = phi_10463_;
                    if _e2828_ {
                        phi_10478_ = true;
                    } else {
                        phi_10478_ = (1f >= _e2796_);
                    }
                    let _e2996_ = phi_10478_;
                    if _e2833_ {
                        phi_10493_ = true;
                    } else {
                        phi_10493_ = (1f >= _e2801_);
                    }
                    let _e3000_ = phi_10493_;
                    let _e3002_ = (_e2805_ - select(_e2786_, 1f, _e2988_));
                    let _e3003_ = (_e2809_ - select(_e2791_, 1f, _e2992_));
                    let _e3004_ = (_e2813_ - select(_e2796_, 1f, _e2996_));
                    let _e3005_ = (_e2817_ - select(_e2801_, 1f, _e3000_));
                    if (_e3002_ != _e3002_) {
                        phi_10508_ = true;
                    } else {
                        phi_10508_ = (0f >= _e3002_);
                    }
                    let _e3009_ = phi_10508_;
                    if (_e3003_ != _e3003_) {
                        phi_10523_ = true;
                    } else {
                        phi_10523_ = (0f >= _e3003_);
                    }
                    let _e3014_ = phi_10523_;
                    if (_e3004_ != _e3004_) {
                        phi_10538_ = true;
                    } else {
                        phi_10538_ = (0f >= _e3004_);
                    }
                    let _e3019_ = phi_10538_;
                    if (_e3005_ != _e3005_) {
                        phi_10553_ = true;
                    } else {
                        phi_10553_ = (0f >= _e3005_);
                    }
                    let _e3024_ = phi_10553_;
                    let _e3028_ = bitcast<f32>((bitcast<u32>(_e2723_) & 2147483647u));
                    let _e3031_ = bitcast<f32>((bitcast<u32>(_e2728_) & 2147483647u));
                    let _e3034_ = bitcast<f32>((bitcast<u32>(_e2733_) & 2147483647u));
                    let _e3037_ = bitcast<f32>((bitcast<u32>(_e2738_) & 2147483647u));
                    let _e3040_ = bitcast<f32>((bitcast<u32>(_e2765_) & 2147483647u));
                    let _e3043_ = bitcast<f32>((bitcast<u32>(_e2770_) & 2147483647u));
                    let _e3046_ = bitcast<f32>((bitcast<u32>(_e2775_) & 2147483647u));
                    let _e3049_ = bitcast<f32>((bitcast<u32>(_e2780_) & 2147483647u));
                    if (_e3028_ != _e3028_) {
                        phi_10568_ = true;
                    } else {
                        phi_10568_ = (_e3040_ >= _e3028_);
                    }
                    let _e3053_ = phi_10568_;
                    let _e3054_ = select(_e3028_, _e3040_, _e3053_);
                    if (_e3031_ != _e3031_) {
                        phi_10583_ = true;
                    } else {
                        phi_10583_ = (_e3043_ >= _e3031_);
                    }
                    let _e3058_ = phi_10583_;
                    let _e3059_ = select(_e3031_, _e3043_, _e3058_);
                    if (_e3034_ != _e3034_) {
                        phi_10598_ = true;
                    } else {
                        phi_10598_ = (_e3046_ >= _e3034_);
                    }
                    let _e3063_ = phi_10598_;
                    let _e3064_ = select(_e3034_, _e3046_, _e3063_);
                    if (_e3037_ != _e3037_) {
                        phi_10613_ = true;
                    } else {
                        phi_10613_ = (_e3049_ >= _e3037_);
                    }
                    let _e3068_ = phi_10613_;
                    let _e3069_ = select(_e3037_, _e3049_, _e3068_);
                    let _e3086_ = (_e3054_ <= 0f);
                    let _e3087_ = select((_e2902_ * (1.5f - (0.5f * (_e2914_ + _e2918_)))), 0f, _e3086_);
                    let _e3088_ = (_e3059_ <= 0f);
                    let _e3089_ = select((_e2903_ * (1.5f - (0.5f * (_e2915_ + _e2919_)))), 0f, _e3088_);
                    let _e3090_ = (_e3064_ <= 0f);
                    let _e3091_ = select((_e2904_ * (1.5f - (0.5f * (_e2916_ + _e2920_)))), 0f, _e3090_);
                    let _e3092_ = (_e3069_ <= 0f);
                    let _e3093_ = select((_e2905_ * (1.5f - (0.5f * (_e2917_ + _e2921_)))), 0f, _e3092_);
                    let _e3135_ = select(((_e2914_ + (0.33333334f * _e2918_)) - (((_e2914_ * _e2914_) + ((2f * _e2914_) * _e2918_)) + (0.2f * (_e2918_ * _e2918_)))), 0f, _e3086_);
                    let _e3136_ = select(((_e2915_ + (0.33333334f * _e2919_)) - (((_e2915_ * _e2915_) + ((2f * _e2915_) * _e2919_)) + (0.2f * (_e2919_ * _e2919_)))), 0f, _e3088_);
                    let _e3137_ = select(((_e2916_ + (0.33333334f * _e2920_)) - (((_e2916_ * _e2916_) + ((2f * _e2916_) * _e2920_)) + (0.2f * (_e2920_ * _e2920_)))), 0f, _e3090_);
                    let _e3138_ = select(((_e2917_ + (0.33333334f * _e2921_)) - (((_e2917_ * _e2917_) + ((2f * _e2917_) * _e2921_)) + (0.2f * (_e2921_ * _e2921_)))), 0f, _e3092_);
                    if (_e3054_ <= 1f) {
                        if (_e3059_ <= 1f) {
                            let _e3142_ = (_e3064_ <= 1f);
                            if _e3142_ {
                                phi_6458_ = (_e3069_ <= 1f);
                            } else {
                                phi_6458_ = bool();
                            }
                            let _e3145_ = phi_6458_;
                            phi_6460_ = _e3145_;
                            phi_6461_ = select(true, false, _e3142_);
                        } else {
                            phi_6460_ = bool();
                            phi_6461_ = true;
                        }
                        let _e3148_ = phi_6460_;
                        let _e3150_ = phi_6461_;
                        phi_6462_ = _e3148_;
                        phi_6463_ = _e3150_;
                    } else {
                        phi_6462_ = bool();
                        phi_6463_ = true;
                    }
                    let _e3152_ = phi_6462_;
                    let _e3154_ = phi_6463_;
                    if select(_e3152_, false, _e3154_) {
                        phi_6841_ = type_27_(type_22_(array<f32, 4>(_e3087_, _e3089_, _e3091_, _e3093_)), type_22_(array<f32, 4>(_e3135_, _e3136_, _e3137_, _e3138_)), type_22_(_e2781_));
                    } else {
                        if (_e2942_ != _e2942_) {
                            phi_10628_ = true;
                        } else {
                            phi_10628_ = (0.000001f >= _e2942_);
                        }
                        let _e3159_ = phi_10628_;
                        let _e3160_ = select(_e2942_, 0.000001f, _e3159_);
                        if (_e2943_ != _e2943_) {
                            phi_10643_ = true;
                        } else {
                            phi_10643_ = (0.000001f >= _e2943_);
                        }
                        let _e3164_ = phi_10643_;
                        let _e3165_ = select(_e2943_, 0.000001f, _e3164_);
                        if (_e2944_ != _e2944_) {
                            phi_10658_ = true;
                        } else {
                            phi_10658_ = (0.000001f >= _e2944_);
                        }
                        let _e3169_ = phi_10658_;
                        let _e3170_ = select(_e2944_, 0.000001f, _e3169_);
                        if (_e2945_ != _e2945_) {
                            phi_10673_ = true;
                        } else {
                            phi_10673_ = (0.000001f >= _e2945_);
                        }
                        let _e3174_ = phi_10673_;
                        let _e3175_ = select(_e2945_, 0.000001f, _e3174_);
                        let _e3178_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3160_)));
                        let _e3181_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3165_)));
                        let _e3184_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3170_)));
                        let _e3187_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3175_)));
                        let _e3197_ = (_e3178_ + (_e3178_ * (1f - (_e3160_ * _e3178_))));
                        let _e3199_ = (_e3181_ + (_e3181_ * (1f - (_e3165_ * _e3181_))));
                        let _e3201_ = (_e3184_ + (_e3184_ * (1f - (_e3170_ * _e3184_))));
                        let _e3203_ = (_e3187_ + (_e3187_ * (1f - (_e3175_ * _e3187_))));
                        let _e3213_ = (_e3197_ + (_e3197_ * (1f - (_e3160_ * _e3197_))));
                        let _e3215_ = (_e3199_ + (_e3199_ * (1f - (_e3165_ * _e3199_))));
                        let _e3217_ = (_e3201_ + (_e3201_ * (1f - (_e3170_ * _e3201_))));
                        let _e3219_ = (_e3203_ + (_e3203_ * (1f - (_e3175_ * _e3203_))));
                        let _e3272_ = (select((select(0f, 1f, (-1f <= _e2786_)) * select(0f, 1f, (_e2805_ <= 1f))), 1f, (_e2942_ <= 0.000001f)) <= 0f);
                        let _e3274_ = (select((select(0f, 1f, (-1f <= _e2791_)) * select(0f, 1f, (_e2809_ <= 1f))), 1f, (_e2943_ <= 0.000001f)) <= 0f);
                        let _e3276_ = (select((select(0f, 1f, (-1f <= _e2796_)) * select(0f, 1f, (_e2813_ <= 1f))), 1f, (_e2944_ <= 0.000001f)) <= 0f);
                        let _e3278_ = (select((select(0f, 1f, (-1f <= _e2801_)) * select(0f, 1f, (_e2817_ <= 1f))), 1f, (_e2945_ <= 0.000001f)) <= 0f);
                        let _e3284_ = (1f <= _e2786_);
                        let _e3286_ = (1f <= _e2791_);
                        let _e3288_ = (1f <= _e2796_);
                        let _e3290_ = (1f <= _e2801_);
                        let _e3292_ = (_e2805_ <= -1f);
                        let _e3294_ = (_e2809_ <= -1f);
                        let _e3296_ = (_e2813_ <= -1f);
                        let _e3298_ = (_e2817_ <= -1f);
                        phi_6841_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3087_, (((select(_e3002_, 0f, _e3009_) - select(_e2962_, 0f, _e2969_)) + (_e2926_ * _e3087_)) * _e3213_), _e3272_), 1f, _e3284_), -1f, _e3292_), select(select(select(_e3089_, (((select(_e3003_, 0f, _e3014_) - select(_e2963_, 0f, _e2974_)) + (_e2931_ * _e3089_)) * _e3215_), _e3274_), 1f, _e3286_), -1f, _e3294_), select(select(select(_e3091_, (((select(_e3004_, 0f, _e3019_) - select(_e2964_, 0f, _e2979_)) + (_e2936_ * _e3091_)) * _e3217_), _e3276_), 1f, _e3288_), -1f, _e3296_), select(select(select(_e3093_, (((select(_e3005_, 0f, _e3024_) - select(_e2965_, 0f, _e2984_)) + (_e2941_ * _e3093_)) * _e3219_), _e3278_), 1f, _e3290_), -1f, _e3298_))), type_22_(array<f32, 4>(select(select(select(_e3135_, ((_e2926_ * _e3135_) * _e3213_), _e3272_), 0f, _e3284_), 0f, _e3292_), select(select(select(_e3136_, ((_e2931_ * _e3136_) * _e3215_), _e3274_), 0f, _e3286_), 0f, _e3294_), select(select(select(_e3137_, ((_e2936_ * _e3137_) * _e3217_), _e3276_), 0f, _e3288_), 0f, _e3296_), select(select(select(_e3138_, ((_e2941_ * _e3138_) * _e3219_), _e3278_), 0f, _e3290_), 0f, _e3298_))), type_22_(_e2781_));
                    }
                    let _e3319_ = phi_6841_;
                    let _e3326_ = select(0f, _e2696_, (bitcast<f32>((bitcast<u32>(_e2696_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3327_ = (_e3326_ != _e3326_);
                    if _e3327_ {
                        phi_10688_ = true;
                    } else {
                        phi_10688_ = (-1000000000000000000f >= _e3326_);
                    }
                    let _e3330_ = phi_10688_;
                    let _e3331_ = select(_e3326_, -1000000000000000000f, _e3330_);
                    if _e3327_ {
                        phi_10703_ = true;
                    } else {
                        phi_10703_ = (-1000000000000000000f >= _e3326_);
                    }
                    let _e3334_ = phi_10703_;
                    let _e3335_ = select(_e3326_, -1000000000000000000f, _e3334_);
                    if _e3327_ {
                        phi_10718_ = true;
                    } else {
                        phi_10718_ = (-1000000000000000000f >= _e3326_);
                    }
                    let _e3338_ = phi_10718_;
                    let _e3339_ = select(_e3326_, -1000000000000000000f, _e3338_);
                    if _e3327_ {
                        phi_10733_ = true;
                    } else {
                        phi_10733_ = (-1000000000000000000f >= _e3326_);
                    }
                    let _e3342_ = phi_10733_;
                    let _e3343_ = select(_e3326_, -1000000000000000000f, _e3342_);
                    if (_e3331_ != _e3331_) {
                        phi_10748_ = true;
                    } else {
                        phi_10748_ = (1000000000000000000f <= _e3331_);
                    }
                    let _e3347_ = phi_10748_;
                    let _e3348_ = select(_e3331_, 1000000000000000000f, _e3347_);
                    if (_e3335_ != _e3335_) {
                        phi_10763_ = true;
                    } else {
                        phi_10763_ = (1000000000000000000f <= _e3335_);
                    }
                    let _e3352_ = phi_10763_;
                    let _e3353_ = select(_e3335_, 1000000000000000000f, _e3352_);
                    if (_e3339_ != _e3339_) {
                        phi_10778_ = true;
                    } else {
                        phi_10778_ = (1000000000000000000f <= _e3339_);
                    }
                    let _e3357_ = phi_10778_;
                    let _e3358_ = select(_e3339_, 1000000000000000000f, _e3357_);
                    if (_e3343_ != _e3343_) {
                        phi_10793_ = true;
                    } else {
                        phi_10793_ = (1000000000000000000f <= _e3343_);
                    }
                    let _e3362_ = phi_10793_;
                    let _e3363_ = select(_e3343_, 1000000000000000000f, _e3362_);
                    let _e3368_ = select(0f, _e2671_, (bitcast<f32>((bitcast<u32>(_e2671_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3369_ = (_e3368_ != _e3368_);
                    if _e3369_ {
                        phi_10808_ = true;
                    } else {
                        phi_10808_ = (-1000000000000000000f >= _e3368_);
                    }
                    let _e3372_ = phi_10808_;
                    let _e3373_ = select(_e3368_, -1000000000000000000f, _e3372_);
                    if _e3369_ {
                        phi_10823_ = true;
                    } else {
                        phi_10823_ = (-1000000000000000000f >= _e3368_);
                    }
                    let _e3376_ = phi_10823_;
                    let _e3377_ = select(_e3368_, -1000000000000000000f, _e3376_);
                    if _e3369_ {
                        phi_10838_ = true;
                    } else {
                        phi_10838_ = (-1000000000000000000f >= _e3368_);
                    }
                    let _e3380_ = phi_10838_;
                    let _e3381_ = select(_e3368_, -1000000000000000000f, _e3380_);
                    if _e3369_ {
                        phi_10853_ = true;
                    } else {
                        phi_10853_ = (-1000000000000000000f >= _e3368_);
                    }
                    let _e3384_ = phi_10853_;
                    let _e3385_ = select(_e3368_, -1000000000000000000f, _e3384_);
                    if (_e3373_ != _e3373_) {
                        phi_10868_ = true;
                    } else {
                        phi_10868_ = (1000000000000000000f <= _e3373_);
                    }
                    let _e3389_ = phi_10868_;
                    let _e3390_ = select(_e3373_, 1000000000000000000f, _e3389_);
                    if (_e3377_ != _e3377_) {
                        phi_10883_ = true;
                    } else {
                        phi_10883_ = (1000000000000000000f <= _e3377_);
                    }
                    let _e3394_ = phi_10883_;
                    let _e3395_ = select(_e3377_, 1000000000000000000f, _e3394_);
                    if (_e3381_ != _e3381_) {
                        phi_10898_ = true;
                    } else {
                        phi_10898_ = (1000000000000000000f <= _e3381_);
                    }
                    let _e3399_ = phi_10898_;
                    let _e3400_ = select(_e3381_, 1000000000000000000f, _e3399_);
                    if (_e3385_ != _e3385_) {
                        phi_10913_ = true;
                    } else {
                        phi_10913_ = (1000000000000000000f <= _e3385_);
                    }
                    let _e3404_ = phi_10913_;
                    let _e3405_ = select(_e3385_, 1000000000000000000f, _e3404_);
                    let _e3406_ = array<f32, 4>(_e3390_, _e3395_, _e3400_, _e3405_);
                    let _e3407_ = (_e3348_ != _e3348_);
                    if _e3407_ {
                        phi_10928_ = true;
                    } else {
                        phi_10928_ = (_e3390_ <= _e3348_);
                    }
                    let _e3410_ = phi_10928_;
                    let _e3411_ = select(_e3348_, _e3390_, _e3410_);
                    let _e3412_ = (_e3353_ != _e3353_);
                    if _e3412_ {
                        phi_10943_ = true;
                    } else {
                        phi_10943_ = (_e3395_ <= _e3353_);
                    }
                    let _e3415_ = phi_10943_;
                    let _e3416_ = select(_e3353_, _e3395_, _e3415_);
                    let _e3417_ = (_e3358_ != _e3358_);
                    if _e3417_ {
                        phi_10958_ = true;
                    } else {
                        phi_10958_ = (_e3400_ <= _e3358_);
                    }
                    let _e3420_ = phi_10958_;
                    let _e3421_ = select(_e3358_, _e3400_, _e3420_);
                    let _e3422_ = (_e3363_ != _e3363_);
                    if _e3422_ {
                        phi_10973_ = true;
                    } else {
                        phi_10973_ = (_e3405_ <= _e3363_);
                    }
                    let _e3425_ = phi_10973_;
                    let _e3426_ = select(_e3363_, _e3405_, _e3425_);
                    if _e3407_ {
                        phi_10988_ = true;
                    } else {
                        phi_10988_ = (_e3390_ >= _e3348_);
                    }
                    let _e3429_ = phi_10988_;
                    let _e3430_ = select(_e3348_, _e3390_, _e3429_);
                    if _e3412_ {
                        phi_11003_ = true;
                    } else {
                        phi_11003_ = (_e3395_ >= _e3353_);
                    }
                    let _e3433_ = phi_11003_;
                    let _e3434_ = select(_e3353_, _e3395_, _e3433_);
                    if _e3417_ {
                        phi_11018_ = true;
                    } else {
                        phi_11018_ = (_e3400_ >= _e3358_);
                    }
                    let _e3437_ = phi_11018_;
                    let _e3438_ = select(_e3358_, _e3400_, _e3437_);
                    if _e3422_ {
                        phi_11033_ = true;
                    } else {
                        phi_11033_ = (_e3405_ >= _e3363_);
                    }
                    let _e3441_ = phi_11033_;
                    let _e3442_ = select(_e3363_, _e3405_, _e3441_);
                    let _e3443_ = (_e3411_ != _e3411_);
                    if _e3443_ {
                        phi_11048_ = true;
                    } else {
                        phi_11048_ = (-1f >= _e3411_);
                    }
                    let _e3446_ = phi_11048_;
                    let _e3447_ = select(_e3411_, -1f, _e3446_);
                    let _e3448_ = (_e3416_ != _e3416_);
                    if _e3448_ {
                        phi_11063_ = true;
                    } else {
                        phi_11063_ = (-1f >= _e3416_);
                    }
                    let _e3451_ = phi_11063_;
                    let _e3452_ = select(_e3416_, -1f, _e3451_);
                    let _e3453_ = (_e3421_ != _e3421_);
                    if _e3453_ {
                        phi_11078_ = true;
                    } else {
                        phi_11078_ = (-1f >= _e3421_);
                    }
                    let _e3456_ = phi_11078_;
                    let _e3457_ = select(_e3421_, -1f, _e3456_);
                    let _e3458_ = (_e3426_ != _e3426_);
                    if _e3458_ {
                        phi_11093_ = true;
                    } else {
                        phi_11093_ = (-1f >= _e3426_);
                    }
                    let _e3461_ = phi_11093_;
                    let _e3462_ = select(_e3426_, -1f, _e3461_);
                    if (_e3447_ != _e3447_) {
                        phi_11108_ = true;
                    } else {
                        phi_11108_ = (1f <= _e3447_);
                    }
                    let _e3466_ = phi_11108_;
                    let _e3467_ = select(_e3447_, 1f, _e3466_);
                    if (_e3452_ != _e3452_) {
                        phi_11123_ = true;
                    } else {
                        phi_11123_ = (1f <= _e3452_);
                    }
                    let _e3471_ = phi_11123_;
                    let _e3472_ = select(_e3452_, 1f, _e3471_);
                    if (_e3457_ != _e3457_) {
                        phi_11138_ = true;
                    } else {
                        phi_11138_ = (1f <= _e3457_);
                    }
                    let _e3476_ = phi_11138_;
                    let _e3477_ = select(_e3457_, 1f, _e3476_);
                    if (_e3462_ != _e3462_) {
                        phi_11153_ = true;
                    } else {
                        phi_11153_ = (1f <= _e3462_);
                    }
                    let _e3481_ = phi_11153_;
                    let _e3482_ = select(_e3462_, 1f, _e3481_);
                    let _e3483_ = (_e3430_ != _e3430_);
                    if _e3483_ {
                        phi_11168_ = true;
                    } else {
                        phi_11168_ = (-1f >= _e3430_);
                    }
                    let _e3486_ = phi_11168_;
                    let _e3487_ = select(_e3430_, -1f, _e3486_);
                    let _e3488_ = (_e3434_ != _e3434_);
                    if _e3488_ {
                        phi_11183_ = true;
                    } else {
                        phi_11183_ = (-1f >= _e3434_);
                    }
                    let _e3491_ = phi_11183_;
                    let _e3492_ = select(_e3434_, -1f, _e3491_);
                    let _e3493_ = (_e3438_ != _e3438_);
                    if _e3493_ {
                        phi_11198_ = true;
                    } else {
                        phi_11198_ = (-1f >= _e3438_);
                    }
                    let _e3496_ = phi_11198_;
                    let _e3497_ = select(_e3438_, -1f, _e3496_);
                    let _e3498_ = (_e3442_ != _e3442_);
                    if _e3498_ {
                        phi_11213_ = true;
                    } else {
                        phi_11213_ = (-1f >= _e3442_);
                    }
                    let _e3501_ = phi_11213_;
                    let _e3502_ = select(_e3442_, -1f, _e3501_);
                    if (_e3487_ != _e3487_) {
                        phi_11228_ = true;
                    } else {
                        phi_11228_ = (1f <= _e3487_);
                    }
                    let _e3506_ = phi_11228_;
                    let _e3507_ = select(_e3487_, 1f, _e3506_);
                    if (_e3492_ != _e3492_) {
                        phi_11243_ = true;
                    } else {
                        phi_11243_ = (1f <= _e3492_);
                    }
                    let _e3511_ = phi_11243_;
                    let _e3512_ = select(_e3492_, 1f, _e3511_);
                    if (_e3497_ != _e3497_) {
                        phi_11258_ = true;
                    } else {
                        phi_11258_ = (1f <= _e3497_);
                    }
                    let _e3516_ = phi_11258_;
                    let _e3517_ = select(_e3497_, 1f, _e3516_);
                    if (_e3502_ != _e3502_) {
                        phi_11273_ = true;
                    } else {
                        phi_11273_ = (1f <= _e3502_);
                    }
                    let _e3521_ = phi_11273_;
                    let _e3522_ = select(_e3502_, 1f, _e3521_);
                    let _e3527_ = (0.5f * (_e3507_ + _e3467_));
                    let _e3528_ = (0.5f * (_e3512_ + _e3472_));
                    let _e3529_ = (0.5f * (_e3517_ + _e3477_));
                    let _e3530_ = (0.5f * (_e3522_ + _e3482_));
                    let _e3531_ = (_e3507_ - _e3467_);
                    let _e3532_ = (_e3512_ - _e3472_);
                    let _e3533_ = (_e3517_ - _e3477_);
                    let _e3534_ = (_e3522_ - _e3482_);
                    let _e3535_ = (0.5f * _e3531_);
                    let _e3536_ = (0.5f * _e3532_);
                    let _e3537_ = (0.5f * _e3533_);
                    let _e3538_ = (0.5f * _e3534_);
                    let _e3539_ = (_e3527_ * _e3527_);
                    let _e3540_ = (_e3528_ * _e3528_);
                    let _e3541_ = (_e3529_ * _e3529_);
                    let _e3542_ = (_e3530_ * _e3530_);
                    let _e3543_ = (_e3535_ * _e3535_);
                    let _e3544_ = (_e3536_ * _e3536_);
                    let _e3545_ = (_e3537_ * _e3537_);
                    let _e3546_ = (_e3538_ * _e3538_);
                    if (_e3531_ != _e3531_) {
                        phi_11288_ = true;
                    } else {
                        phi_11288_ = (0f >= _e3531_);
                    }
                    let _e3550_ = phi_11288_;
                    let _e3551_ = select(_e3531_, 0f, _e3550_);
                    if (_e3532_ != _e3532_) {
                        phi_11303_ = true;
                    } else {
                        phi_11303_ = (0f >= _e3532_);
                    }
                    let _e3555_ = phi_11303_;
                    let _e3556_ = select(_e3532_, 0f, _e3555_);
                    if (_e3533_ != _e3533_) {
                        phi_11318_ = true;
                    } else {
                        phi_11318_ = (0f >= _e3533_);
                    }
                    let _e3560_ = phi_11318_;
                    let _e3561_ = select(_e3533_, 0f, _e3560_);
                    if (_e3534_ != _e3534_) {
                        phi_11333_ = true;
                    } else {
                        phi_11333_ = (0f >= _e3534_);
                    }
                    let _e3565_ = phi_11333_;
                    let _e3566_ = select(_e3534_, 0f, _e3565_);
                    let _e3567_ = (_e3430_ - _e3411_);
                    let _e3568_ = (_e3434_ - _e3416_);
                    let _e3569_ = (_e3438_ - _e3421_);
                    let _e3570_ = (_e3442_ - _e3426_);
                    if _e3483_ {
                        phi_11348_ = true;
                    } else {
                        phi_11348_ = (-1f <= _e3430_);
                    }
                    let _e3573_ = phi_11348_;
                    if _e3488_ {
                        phi_11363_ = true;
                    } else {
                        phi_11363_ = (-1f <= _e3434_);
                    }
                    let _e3577_ = phi_11363_;
                    if _e3493_ {
                        phi_11378_ = true;
                    } else {
                        phi_11378_ = (-1f <= _e3438_);
                    }
                    let _e3581_ = phi_11378_;
                    if _e3498_ {
                        phi_11393_ = true;
                    } else {
                        phi_11393_ = (-1f <= _e3442_);
                    }
                    let _e3585_ = phi_11393_;
                    let _e3587_ = (select(_e3430_, -1f, _e3573_) - _e3411_);
                    let _e3588_ = (select(_e3434_, -1f, _e3577_) - _e3416_);
                    let _e3589_ = (select(_e3438_, -1f, _e3581_) - _e3421_);
                    let _e3590_ = (select(_e3442_, -1f, _e3585_) - _e3426_);
                    if (_e3587_ != _e3587_) {
                        phi_11408_ = true;
                    } else {
                        phi_11408_ = (0f >= _e3587_);
                    }
                    let _e3594_ = phi_11408_;
                    if (_e3588_ != _e3588_) {
                        phi_11423_ = true;
                    } else {
                        phi_11423_ = (0f >= _e3588_);
                    }
                    let _e3599_ = phi_11423_;
                    if (_e3589_ != _e3589_) {
                        phi_11438_ = true;
                    } else {
                        phi_11438_ = (0f >= _e3589_);
                    }
                    let _e3604_ = phi_11438_;
                    if (_e3590_ != _e3590_) {
                        phi_11453_ = true;
                    } else {
                        phi_11453_ = (0f >= _e3590_);
                    }
                    let _e3609_ = phi_11453_;
                    if _e3443_ {
                        phi_11468_ = true;
                    } else {
                        phi_11468_ = (1f >= _e3411_);
                    }
                    let _e3613_ = phi_11468_;
                    if _e3448_ {
                        phi_11483_ = true;
                    } else {
                        phi_11483_ = (1f >= _e3416_);
                    }
                    let _e3617_ = phi_11483_;
                    if _e3453_ {
                        phi_11498_ = true;
                    } else {
                        phi_11498_ = (1f >= _e3421_);
                    }
                    let _e3621_ = phi_11498_;
                    if _e3458_ {
                        phi_11513_ = true;
                    } else {
                        phi_11513_ = (1f >= _e3426_);
                    }
                    let _e3625_ = phi_11513_;
                    let _e3627_ = (_e3430_ - select(_e3411_, 1f, _e3613_));
                    let _e3628_ = (_e3434_ - select(_e3416_, 1f, _e3617_));
                    let _e3629_ = (_e3438_ - select(_e3421_, 1f, _e3621_));
                    let _e3630_ = (_e3442_ - select(_e3426_, 1f, _e3625_));
                    if (_e3627_ != _e3627_) {
                        phi_11528_ = true;
                    } else {
                        phi_11528_ = (0f >= _e3627_);
                    }
                    let _e3634_ = phi_11528_;
                    if (_e3628_ != _e3628_) {
                        phi_11543_ = true;
                    } else {
                        phi_11543_ = (0f >= _e3628_);
                    }
                    let _e3639_ = phi_11543_;
                    if (_e3629_ != _e3629_) {
                        phi_11558_ = true;
                    } else {
                        phi_11558_ = (0f >= _e3629_);
                    }
                    let _e3644_ = phi_11558_;
                    if (_e3630_ != _e3630_) {
                        phi_11573_ = true;
                    } else {
                        phi_11573_ = (0f >= _e3630_);
                    }
                    let _e3649_ = phi_11573_;
                    let _e3653_ = bitcast<f32>((bitcast<u32>(_e3348_) & 2147483647u));
                    let _e3656_ = bitcast<f32>((bitcast<u32>(_e3353_) & 2147483647u));
                    let _e3659_ = bitcast<f32>((bitcast<u32>(_e3358_) & 2147483647u));
                    let _e3662_ = bitcast<f32>((bitcast<u32>(_e3363_) & 2147483647u));
                    let _e3665_ = bitcast<f32>((bitcast<u32>(_e3390_) & 2147483647u));
                    let _e3668_ = bitcast<f32>((bitcast<u32>(_e3395_) & 2147483647u));
                    let _e3671_ = bitcast<f32>((bitcast<u32>(_e3400_) & 2147483647u));
                    let _e3674_ = bitcast<f32>((bitcast<u32>(_e3405_) & 2147483647u));
                    if (_e3653_ != _e3653_) {
                        phi_11588_ = true;
                    } else {
                        phi_11588_ = (_e3665_ >= _e3653_);
                    }
                    let _e3678_ = phi_11588_;
                    let _e3679_ = select(_e3653_, _e3665_, _e3678_);
                    if (_e3656_ != _e3656_) {
                        phi_11603_ = true;
                    } else {
                        phi_11603_ = (_e3668_ >= _e3656_);
                    }
                    let _e3683_ = phi_11603_;
                    let _e3684_ = select(_e3656_, _e3668_, _e3683_);
                    if (_e3659_ != _e3659_) {
                        phi_11618_ = true;
                    } else {
                        phi_11618_ = (_e3671_ >= _e3659_);
                    }
                    let _e3688_ = phi_11618_;
                    let _e3689_ = select(_e3659_, _e3671_, _e3688_);
                    if (_e3662_ != _e3662_) {
                        phi_11633_ = true;
                    } else {
                        phi_11633_ = (_e3674_ >= _e3662_);
                    }
                    let _e3693_ = phi_11633_;
                    let _e3694_ = select(_e3662_, _e3674_, _e3693_);
                    let _e3711_ = (_e3679_ <= 0f);
                    let _e3712_ = select((_e3527_ * (1.5f - (0.5f * (_e3539_ + _e3543_)))), 0f, _e3711_);
                    let _e3713_ = (_e3684_ <= 0f);
                    let _e3714_ = select((_e3528_ * (1.5f - (0.5f * (_e3540_ + _e3544_)))), 0f, _e3713_);
                    let _e3715_ = (_e3689_ <= 0f);
                    let _e3716_ = select((_e3529_ * (1.5f - (0.5f * (_e3541_ + _e3545_)))), 0f, _e3715_);
                    let _e3717_ = (_e3694_ <= 0f);
                    let _e3718_ = select((_e3530_ * (1.5f - (0.5f * (_e3542_ + _e3546_)))), 0f, _e3717_);
                    let _e3760_ = select(((_e3539_ + (0.33333334f * _e3543_)) - (((_e3539_ * _e3539_) + ((2f * _e3539_) * _e3543_)) + (0.2f * (_e3543_ * _e3543_)))), 0f, _e3711_);
                    let _e3761_ = select(((_e3540_ + (0.33333334f * _e3544_)) - (((_e3540_ * _e3540_) + ((2f * _e3540_) * _e3544_)) + (0.2f * (_e3544_ * _e3544_)))), 0f, _e3713_);
                    let _e3762_ = select(((_e3541_ + (0.33333334f * _e3545_)) - (((_e3541_ * _e3541_) + ((2f * _e3541_) * _e3545_)) + (0.2f * (_e3545_ * _e3545_)))), 0f, _e3715_);
                    let _e3763_ = select(((_e3542_ + (0.33333334f * _e3546_)) - (((_e3542_ * _e3542_) + ((2f * _e3542_) * _e3546_)) + (0.2f * (_e3546_ * _e3546_)))), 0f, _e3717_);
                    if (_e3679_ <= 1f) {
                        if (_e3684_ <= 1f) {
                            let _e3767_ = (_e3689_ <= 1f);
                            if _e3767_ {
                                phi_7265_ = (_e3694_ <= 1f);
                            } else {
                                phi_7265_ = bool();
                            }
                            let _e3770_ = phi_7265_;
                            phi_7267_ = _e3770_;
                            phi_7268_ = select(true, false, _e3767_);
                        } else {
                            phi_7267_ = bool();
                            phi_7268_ = true;
                        }
                        let _e3773_ = phi_7267_;
                        let _e3775_ = phi_7268_;
                        phi_7269_ = _e3773_;
                        phi_7270_ = _e3775_;
                    } else {
                        phi_7269_ = bool();
                        phi_7270_ = true;
                    }
                    let _e3777_ = phi_7269_;
                    let _e3779_ = phi_7270_;
                    if select(_e3777_, false, _e3779_) {
                        phi_7648_ = type_27_(type_22_(array<f32, 4>(_e3712_, _e3714_, _e3716_, _e3718_)), type_22_(array<f32, 4>(_e3760_, _e3761_, _e3762_, _e3763_)), type_22_(_e3406_));
                    } else {
                        if (_e3567_ != _e3567_) {
                            phi_11648_ = true;
                        } else {
                            phi_11648_ = (0.000001f >= _e3567_);
                        }
                        let _e3784_ = phi_11648_;
                        let _e3785_ = select(_e3567_, 0.000001f, _e3784_);
                        if (_e3568_ != _e3568_) {
                            phi_11663_ = true;
                        } else {
                            phi_11663_ = (0.000001f >= _e3568_);
                        }
                        let _e3789_ = phi_11663_;
                        let _e3790_ = select(_e3568_, 0.000001f, _e3789_);
                        if (_e3569_ != _e3569_) {
                            phi_11678_ = true;
                        } else {
                            phi_11678_ = (0.000001f >= _e3569_);
                        }
                        let _e3794_ = phi_11678_;
                        let _e3795_ = select(_e3569_, 0.000001f, _e3794_);
                        if (_e3570_ != _e3570_) {
                            phi_11693_ = true;
                        } else {
                            phi_11693_ = (0.000001f >= _e3570_);
                        }
                        let _e3799_ = phi_11693_;
                        let _e3800_ = select(_e3570_, 0.000001f, _e3799_);
                        let _e3803_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3785_)));
                        let _e3806_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3790_)));
                        let _e3809_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3795_)));
                        let _e3812_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3800_)));
                        let _e3822_ = (_e3803_ + (_e3803_ * (1f - (_e3785_ * _e3803_))));
                        let _e3824_ = (_e3806_ + (_e3806_ * (1f - (_e3790_ * _e3806_))));
                        let _e3826_ = (_e3809_ + (_e3809_ * (1f - (_e3795_ * _e3809_))));
                        let _e3828_ = (_e3812_ + (_e3812_ * (1f - (_e3800_ * _e3812_))));
                        let _e3838_ = (_e3822_ + (_e3822_ * (1f - (_e3785_ * _e3822_))));
                        let _e3840_ = (_e3824_ + (_e3824_ * (1f - (_e3790_ * _e3824_))));
                        let _e3842_ = (_e3826_ + (_e3826_ * (1f - (_e3795_ * _e3826_))));
                        let _e3844_ = (_e3828_ + (_e3828_ * (1f - (_e3800_ * _e3828_))));
                        let _e3897_ = (select((select(0f, 1f, (-1f <= _e3411_)) * select(0f, 1f, (_e3430_ <= 1f))), 1f, (_e3567_ <= 0.000001f)) <= 0f);
                        let _e3899_ = (select((select(0f, 1f, (-1f <= _e3416_)) * select(0f, 1f, (_e3434_ <= 1f))), 1f, (_e3568_ <= 0.000001f)) <= 0f);
                        let _e3901_ = (select((select(0f, 1f, (-1f <= _e3421_)) * select(0f, 1f, (_e3438_ <= 1f))), 1f, (_e3569_ <= 0.000001f)) <= 0f);
                        let _e3903_ = (select((select(0f, 1f, (-1f <= _e3426_)) * select(0f, 1f, (_e3442_ <= 1f))), 1f, (_e3570_ <= 0.000001f)) <= 0f);
                        let _e3909_ = (1f <= _e3411_);
                        let _e3911_ = (1f <= _e3416_);
                        let _e3913_ = (1f <= _e3421_);
                        let _e3915_ = (1f <= _e3426_);
                        let _e3917_ = (_e3430_ <= -1f);
                        let _e3919_ = (_e3434_ <= -1f);
                        let _e3921_ = (_e3438_ <= -1f);
                        let _e3923_ = (_e3442_ <= -1f);
                        phi_7648_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3712_, (((select(_e3627_, 0f, _e3634_) - select(_e3587_, 0f, _e3594_)) + (_e3551_ * _e3712_)) * _e3838_), _e3897_), 1f, _e3909_), -1f, _e3917_), select(select(select(_e3714_, (((select(_e3628_, 0f, _e3639_) - select(_e3588_, 0f, _e3599_)) + (_e3556_ * _e3714_)) * _e3840_), _e3899_), 1f, _e3911_), -1f, _e3919_), select(select(select(_e3716_, (((select(_e3629_, 0f, _e3644_) - select(_e3589_, 0f, _e3604_)) + (_e3561_ * _e3716_)) * _e3842_), _e3901_), 1f, _e3913_), -1f, _e3921_), select(select(select(_e3718_, (((select(_e3630_, 0f, _e3649_) - select(_e3590_, 0f, _e3609_)) + (_e3566_ * _e3718_)) * _e3844_), _e3903_), 1f, _e3915_), -1f, _e3923_))), type_22_(array<f32, 4>(select(select(select(_e3760_, ((_e3551_ * _e3760_) * _e3838_), _e3897_), 0f, _e3909_), 0f, _e3917_), select(select(select(_e3761_, ((_e3556_ * _e3761_) * _e3840_), _e3899_), 0f, _e3911_), 0f, _e3919_), select(select(select(_e3762_, ((_e3561_ * _e3762_) * _e3842_), _e3901_), 0f, _e3913_), 0f, _e3921_), select(select(select(_e3763_, ((_e3566_ * _e3763_) * _e3844_), _e3903_), 0f, _e3915_), 0f, _e3923_))), type_22_(_e3406_));
                    }
                    let _e3944_ = phi_7648_;
                    if _e160_ {
                        phi_7657_ = (_e203_ + _e3944_.member.member[0]);
                        phi_7658_ = (_e205_ + _e3319_.member.member[0]);
                    } else {
                        phi_7657_ = _e203_;
                        phi_7658_ = _e205_;
                    }
                    let _e3952_ = phi_7657_;
                    let _e3954_ = phi_7658_;
                    workgroupBarrier();
                    phi_12455_ = _e2623_;
                    phi_308_ = _e3952_;
                    phi_310_ = _e3954_;
                } else {
                    phi_12455_ = _e199_;
                    phi_308_ = f32();
                    phi_310_ = f32();
                }
                let _e3956_ = phi_12455_;
                let _e3958_ = phi_308_;
                let _e3960_ = phi_310_;
                phi_12014_ = _e3956_;
                phi_304_ = _e218_;
                phi_307_ = _e3958_;
                phi_309_ = _e3960_;
                phi_12454_ = _e3956_;
                if !(_e224_) {
                    break;
                }
            }
            let _e3963_ = phi_12454_;
            if _e3963_ {
                break;
            }
            if _e160_ {
                if (_e119_ == 0u) {
                    phi_7672_ = 0u;
                } else {
                    let _e3965_ = (_e119_ - 1u);
                    phi_7672_ = select(_e3965_, _e162_, (_e162_ < _e3965_));
                }
                let _e3969_ = phi_7672_;
                if (_e3969_ < _e119_) {
                } else {
                    break;
                }
                let _e3974_ = target_audio.member[_e3969_].x;
                let _e3978_ = target_audio.member[_e3969_].y;
                let _e3979_ = bitcast<u32>(_e3974_);
                let _e3980_ = (_e3979_ & 2139095040u);
                if (_e3980_ == 2139095040u) {
                    phi_7695_ = true;
                } else {
                    phi_7695_ = (_e3980_ == 0u);
                }
                let _e3984_ = phi_7695_;
                let _e3987_ = bitcast<u32>(_e3978_);
                let _e3988_ = (_e3987_ & 2139095040u);
                if (_e3988_ == 2139095040u) {
                    phi_7711_ = true;
                } else {
                    phi_7711_ = (_e3988_ == 0u);
                }
                let _e3992_ = phi_7711_;
                phi_7719_ = vec2<f32>(bitcast<f32>(select(_e3979_, 0u, _e3984_)), bitcast<f32>(select(_e3987_, 0u, _e3992_)));
            } else {
                phi_7719_ = vec2<f32>(0f, 0f);
            }
            let _e3997_ = phi_7719_;
            if _e160_ {
                let _e4000_ = local;
                phi_7725_ = (_e4000_ - _e3997_.x);
            } else {
                phi_7725_ = 0f;
            }
            let _e4003_ = phi_7725_;
            if _e160_ {
                let _e4006_ = local_1_;
                phi_7731_ = (_e4006_ - _e3997_.y);
            } else {
                phi_7731_ = 0f;
            }
            let _e4009_ = phi_7731_;
            if _e160_ {
                let _e4013_ = bitcast<u32>(((_e4003_ * _e4003_) + (_e4009_ * _e4009_)));
                let _e4014_ = (_e4013_ & 2139095040u);
                if (_e4014_ == 2139095040u) {
                    phi_7748_ = true;
                } else {
                    phi_7748_ = (_e4014_ == 0u);
                }
                let _e4018_ = phi_7748_;
                phi_7754_ = bitcast<f32>(select(_e4013_, 0u, _e4018_));
            } else {
                phi_7754_ = 0f;
            }
            let _e4022_ = phi_7754_;
            if _e160_ {
                let _e4028_ = bitcast<u32>(((_e3997_.x * _e3997_.x) + (_e3997_.y * _e3997_.y)));
                let _e4029_ = (_e4028_ & 2139095040u);
                if (_e4029_ == 2139095040u) {
                    phi_7775_ = true;
                } else {
                    phi_7775_ = (_e4029_ == 0u);
                }
                let _e4033_ = phi_7775_;
                phi_7781_ = bitcast<f32>(select(_e4028_, 0u, _e4033_));
            } else {
                phi_7781_ = 0f;
            }
            let _e4037_ = phi_7781_;
            let _e4038_ = bitcast<u32>(_e4022_);
            let _e4039_ = (_e4038_ & 2139095040u);
            if (_e4039_ == 2139095040u) {
                phi_7792_ = true;
            } else {
                phi_7792_ = (_e4039_ == 0u);
            }
            let _e4043_ = phi_7792_;
            let _e4046_ = bitcast<u32>(_e4037_);
            let _e4047_ = (_e4046_ & 2139095040u);
            if (_e4047_ == 2139095040u) {
                phi_7808_ = true;
            } else {
                phi_7808_ = (_e4047_ == 0u);
            }
            let _e4051_ = phi_7808_;
            let _e6167 = subgroupAdd(bitcast<f32>(select(_e4038_, 0u, _e4043_)));
            let _e6171 = subgroupAdd(bitcast<f32>(select(_e4046_, 0u, _e4051_)));
            let _e4056_ = (_e115_ == 0u);
            let _e4060_ = select(255u, _e114_, (_e114_ < 255u));
            let _e4061_ = (_e4060_ < 256u);
            if _e4061_ {
            } else {
                break;
            }
            err_scratch[_e4060_] = select(0f, _e6167, _e4056_);
            if _e4061_ {
            } else {
                break;
            }
            tgt_scratch[_e4060_] = select(0f, _e6171, _e4056_);
            workgroupBarrier();
            phi_7846_ = 128u;
            loop {
                let _e4065_ = phi_7846_;
                let _e4066_ = (_e4065_ > 0u);
                if _e4066_ {
                    if (_e114_ < _e4065_) {
                        let _e4068_ = (_e114_ + _e4065_);
                        let _e4069_ = (_e4068_ < 256u);
                        if _e4069_ {
                        } else {
                            phi_12862_ = true;
                            break;
                        }
                        let _e4071_ = err_scratch[_e4068_];
                        let _e4072_ = (_e114_ < 256u);
                        if _e4072_ {
                        } else {
                            phi_12862_ = true;
                            break;
                        }
                        let _e4074_ = err_scratch[_e114_];
                        err_scratch[_e114_] = (_e4074_ + _e4071_);
                        if _e4069_ {
                        } else {
                            phi_12862_ = true;
                            break;
                        }
                        let _e4077_ = tgt_scratch[_e4068_];
                        if _e4072_ {
                        } else {
                            phi_12862_ = true;
                            break;
                        }
                        let _e4079_ = tgt_scratch[_e114_];
                        tgt_scratch[_e114_] = (_e4079_ + _e4077_);
                    }
                    workgroupBarrier();
                    phi_7847_ = (_e4065_ >> bitcast<u32>(1i));
                } else {
                    phi_7847_ = u32();
                }
                let _e4084_ = phi_7847_;
                phi_7846_ = _e4084_;
                phi_12862_ = _e3963_;
                if !(_e4066_) {
                    break;
                }
            }
            let _e4087_ = phi_12862_;
            if _e4087_ {
                break;
            }
            let _e4089_ = err_scratch[0];
            let _e4090_ = bitcast<u32>(_e4089_);
            let _e4091_ = (_e4090_ & 2139095040u);
            if (_e4091_ == 2139095040u) {
                phi_7907_ = true;
            } else {
                phi_7907_ = (_e4091_ == 0u);
            }
            let _e4095_ = phi_7907_;
            let _e4099_ = tgt_scratch[0];
            let _e4100_ = bitcast<u32>(_e4099_);
            let _e4101_ = (_e4100_ & 2139095040u);
            if (_e4101_ == 2139095040u) {
                phi_7925_ = true;
            } else {
                phi_7925_ = (_e4101_ == 0u);
            }
            let _e4105_ = phi_7925_;
            workgroupBarrier();
            if (_e114_ == 0u) {
                if (_e113_.x < _e131_) {
                    if (_e113_.y < _e151_) {
                        if (_e113_.z < select(1u, _e142_, (1u < _e142_))) {
                            let _e4114_ = uniforms.member.member_5_;
                            if ((bitcast<u32>(_e4114_) & 2139095040u) != 2139095040u) {
                                let _e4118_ = (_e4114_ > 0.000000000001f);
                                phi_7965_ = select(f32(), _e4114_, _e4118_);
                                phi_7966_ = select(true, false, _e4118_);
                            } else {
                                phi_7965_ = f32();
                                phi_7966_ = true;
                            }
                            let _e4122_ = phi_7965_;
                            let _e4124_ = phi_7966_;
                            let _e4126_ = (bitcast<f32>(select(_e4100_, 0u, _e4105_)) + select(_e4122_, 0.000000000001f, _e4124_));
                            if ((bitcast<u32>(_e4126_) & 2139095040u) != 2139095040u) {
                                let _e4130_ = (_e4126_ <= 0f);
                                if _e4130_ {
                                    phi_8064_ = f32();
                                } else {
                                    if (_e4126_ < 0.00001f) {
                                        let _e4162_ = (select(0.000000000000000000000001f, _e4126_, (_e4126_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                        if (_e4162_ != _e4162_) {
                                            phi_11768_ = true;
                                        } else {
                                            phi_11768_ = (0.000001f >= _e4162_);
                                        }
                                        let _e4166_ = phi_11768_;
                                        let _e4167_ = select(_e4162_, 0.000001f, _e4166_);
                                        let _e4170_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4167_)));
                                        let _e4174_ = (_e4170_ + (_e4170_ * (1f - (_e4167_ * _e4170_))));
                                        let _e4178_ = (_e4174_ + (_e4174_ * (1f - (_e4167_ * _e4174_))));
                                        let _e4184_ = bitcast<u32>(((_e4178_ + (_e4178_ * (1f - (_e4167_ * _e4178_)))) * 18446744000000000000f));
                                        let _e4185_ = (_e4184_ & 2139095040u);
                                        if (_e4185_ == 2139095040u) {
                                            phi_8022_ = true;
                                        } else {
                                            phi_8022_ = (_e4185_ == 0u);
                                        }
                                        let _e4189_ = phi_8022_;
                                        phi_8063_ = bitcast<f32>(select(_e4184_, 0u, _e4189_));
                                    } else {
                                        if (_e4126_ != _e4126_) {
                                            phi_11708_ = true;
                                        } else {
                                            phi_11708_ = (0.000001f >= _e4126_);
                                        }
                                        let _e4135_ = phi_11708_;
                                        let _e4136_ = select(_e4126_, 0.000001f, _e4135_);
                                        let _e4139_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4136_)));
                                        let _e4143_ = (_e4139_ + (_e4139_ * (1f - (_e4136_ * _e4139_))));
                                        let _e4147_ = (_e4143_ + (_e4143_ * (1f - (_e4136_ * _e4143_))));
                                        let _e4152_ = bitcast<u32>((_e4147_ + (_e4147_ * (1f - (_e4136_ * _e4147_)))));
                                        let _e4153_ = (_e4152_ & 2139095040u);
                                        if (_e4153_ == 2139095040u) {
                                            phi_8057_ = true;
                                        } else {
                                            phi_8057_ = (_e4153_ == 0u);
                                        }
                                        let _e4157_ = phi_8057_;
                                        phi_8063_ = bitcast<f32>(select(_e4152_, 0u, _e4157_));
                                    }
                                    let _e4193_ = phi_8063_;
                                    phi_8064_ = _e4193_;
                                }
                                let _e4195_ = phi_8064_;
                                phi_8065_ = _e4195_;
                                phi_8066_ = _e4130_;
                            } else {
                                phi_8065_ = f32();
                                phi_8066_ = true;
                            }
                            let _e4197_ = phi_8065_;
                            let _e4199_ = phi_8066_;
                            let _e4202_ = bitcast<u32>((bitcast<f32>(select(_e4090_, 0u, _e4095_)) * select(_e4197_, 0f, _e4199_)));
                            let _e4203_ = (_e4202_ & 2139095040u);
                            if (_e4203_ == 2139095040u) {
                                phi_8082_ = true;
                            } else {
                                phi_8082_ = (_e4203_ == 0u);
                            }
                            let _e4207_ = phi_8082_;
                            let _e4213_ = ((((_e113_.z * _e131_) + _e113_.x) * _e151_) + _e113_.y);
                            if (_e123_ == 0u) {
                                phi_8102_ = 0u;
                            } else {
                                let _e4215_ = (_e123_ - 1u);
                                phi_8102_ = select(_e4215_, _e4213_, (_e4213_ < _e4215_));
                            }
                            let _e4219_ = phi_8102_;
                            if (_e4219_ < _e123_) {
                            } else {
                                break;
                            }
                            seed_losses.member[_e4219_] = bitcast<f32>(select(_e4202_, 0u, _e4207_));
                        }
                    }
                }
            }
            break;
        }
    }
    return;
}

@compute @workgroup_size(256, 1, 1) 
fn prescreen_population(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_invocation_id) sg_lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    sg_lid_1_ = sg_lid;
    function_();
    return;
}
