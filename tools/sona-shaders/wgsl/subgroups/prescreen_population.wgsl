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
    var phi_11982_: bool;
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
    var phi_8106_: bool;
    var phi_602_: bool;
    var phi_608_: f32;
    var phi_609_: f32;
    var phi_528_: f32;
    var phi_530_: f32;
    var phi_531_: bool;
    var phi_551_: f32;
    var phi_610_: f32;
    var phi_611_: f32;
    var phi_8166_: bool;
    var phi_697_: bool;
    var phi_8226_: bool;
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
    var phi_8286_: bool;
    var phi_8346_: bool;
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
    var phi_11786_: bool;
    var phi_2565_: u32;
    var phi_2566_: u32;
    var phi_12087_: bool;
    var phi_3108_: bool;
    var phi_3124_: bool;
    var phi_3132_: type_20_;
    var phi_3155_: type_23_;
    var phi_3215_: u32;
    var phi_3216_: u32;
    var phi_12102_: bool;
    var phi_3478_: u32;
    var phi_3479_: u32;
    var phi_12122_: bool;
    var phi_4021_: bool;
    var phi_4037_: bool;
    var phi_4045_: type_20_;
    var phi_4110_: type_15_;
    var phi_4113_: type_22_;
    var phi_4115_: type_22_;
    var phi_4111_: type_15_;
    var phi_4138_: type_15_;
    var phi_4192_: u32;
    var phi_4193_: u32;
    var phi_4206_: u32;
    var phi_4207_: u32;
    var phi_4220_: u32;
    var phi_4221_: u32;
    var phi_4234_: u32;
    var phi_4235_: u32;
    var phi_8406_: bool;
    var phi_8421_: bool;
    var phi_8436_: bool;
    var phi_8451_: bool;
    var phi_8466_: bool;
    var phi_8481_: bool;
    var phi_8496_: bool;
    var phi_8511_: bool;
    var phi_8526_: bool;
    var phi_8541_: bool;
    var phi_8556_: bool;
    var phi_8571_: bool;
    var phi_8586_: bool;
    var phi_8601_: bool;
    var phi_8616_: bool;
    var phi_8631_: bool;
    var phi_8646_: bool;
    var phi_8661_: bool;
    var phi_8676_: bool;
    var phi_8691_: bool;
    var phi_8706_: bool;
    var phi_8721_: bool;
    var phi_8736_: bool;
    var phi_8751_: bool;
    var phi_8766_: bool;
    var phi_8781_: bool;
    var phi_8796_: bool;
    var phi_8811_: bool;
    var phi_8826_: bool;
    var phi_8841_: bool;
    var phi_8856_: bool;
    var phi_8871_: bool;
    var phi_4940_: u32;
    var phi_4941_: u32;
    var phi_4954_: u32;
    var phi_4955_: u32;
    var phi_4968_: u32;
    var phi_4969_: u32;
    var phi_4982_: u32;
    var phi_4983_: u32;
    var phi_5002_: u32;
    var phi_5003_: u32;
    var phi_5016_: u32;
    var phi_5017_: u32;
    var phi_5030_: u32;
    var phi_5031_: u32;
    var phi_5044_: u32;
    var phi_5045_: u32;
    var phi_9006_: bool;
    var phi_9021_: bool;
    var phi_9036_: bool;
    var phi_9051_: bool;
    var phi_9066_: bool;
    var phi_9081_: bool;
    var phi_9096_: bool;
    var phi_9111_: bool;
    var phi_9126_: bool;
    var phi_9141_: bool;
    var phi_9156_: bool;
    var phi_9171_: bool;
    var phi_9186_: bool;
    var phi_9201_: bool;
    var phi_9216_: bool;
    var phi_9231_: bool;
    var phi_9246_: bool;
    var phi_9261_: bool;
    var phi_9276_: bool;
    var phi_9291_: bool;
    var phi_9306_: bool;
    var phi_9321_: bool;
    var phi_9336_: bool;
    var phi_9351_: bool;
    var phi_9366_: bool;
    var phi_9381_: bool;
    var phi_9396_: bool;
    var phi_9411_: bool;
    var phi_9426_: bool;
    var phi_9441_: bool;
    var phi_9456_: bool;
    var phi_9471_: bool;
    var phi_5750_: u32;
    var phi_5751_: u32;
    var phi_5764_: u32;
    var phi_5765_: u32;
    var phi_5778_: u32;
    var phi_5779_: u32;
    var phi_5792_: u32;
    var phi_5793_: u32;
    var phi_4114_: type_22_;
    var phi_4116_: type_22_;
    var phi_12137_: bool;
    var phi_12819_: bool;
    var phi_5877_: f32;
    var phi_5889_: f32;
    var phi_9606_: bool;
    var phi_9621_: bool;
    var phi_5913_: bool;
    var phi_5919_: f32;
    var phi_5937_: bool;
    var phi_5943_: f32;
    var phi_5972_: f32;
    var phi_5974_: f32;
    var phi_5975_: bool;
    var phi_5996_: f32;
    var phi_5998_: f32;
    var phi_5999_: bool;
    var phi_9636_: bool;
    var phi_9651_: bool;
    var phi_9666_: bool;
    var phi_9681_: bool;
    var phi_9696_: bool;
    var phi_9711_: bool;
    var phi_9726_: bool;
    var phi_9741_: bool;
    var phi_9756_: bool;
    var phi_9771_: bool;
    var phi_9786_: bool;
    var phi_9801_: bool;
    var phi_9816_: bool;
    var phi_9831_: bool;
    var phi_9846_: bool;
    var phi_9861_: bool;
    var phi_9876_: bool;
    var phi_9891_: bool;
    var phi_9906_: bool;
    var phi_9921_: bool;
    var phi_9936_: bool;
    var phi_9951_: bool;
    var phi_9966_: bool;
    var phi_9981_: bool;
    var phi_9996_: bool;
    var phi_10011_: bool;
    var phi_10026_: bool;
    var phi_10041_: bool;
    var phi_10056_: bool;
    var phi_10071_: bool;
    var phi_10086_: bool;
    var phi_10101_: bool;
    var phi_10116_: bool;
    var phi_10131_: bool;
    var phi_10146_: bool;
    var phi_10161_: bool;
    var phi_10176_: bool;
    var phi_10191_: bool;
    var phi_10206_: bool;
    var phi_10221_: bool;
    var phi_10236_: bool;
    var phi_10251_: bool;
    var phi_10266_: bool;
    var phi_10281_: bool;
    var phi_10296_: bool;
    var phi_10311_: bool;
    var phi_10326_: bool;
    var phi_10341_: bool;
    var phi_10356_: bool;
    var phi_10371_: bool;
    var phi_10386_: bool;
    var phi_10401_: bool;
    var phi_10416_: bool;
    var phi_10431_: bool;
    var phi_10446_: bool;
    var phi_10461_: bool;
    var phi_10476_: bool;
    var phi_10491_: bool;
    var phi_10506_: bool;
    var phi_10521_: bool;
    var phi_10536_: bool;
    var phi_10551_: bool;
    var phi_10566_: bool;
    var phi_10581_: bool;
    var phi_6426_: bool;
    var phi_6428_: bool;
    var phi_6429_: bool;
    var phi_6430_: bool;
    var phi_6431_: bool;
    var phi_10596_: bool;
    var phi_10611_: bool;
    var phi_10626_: bool;
    var phi_10641_: bool;
    var phi_6809_: type_27_;
    var phi_10656_: bool;
    var phi_10671_: bool;
    var phi_10686_: bool;
    var phi_10701_: bool;
    var phi_10716_: bool;
    var phi_10731_: bool;
    var phi_10746_: bool;
    var phi_10761_: bool;
    var phi_10776_: bool;
    var phi_10791_: bool;
    var phi_10806_: bool;
    var phi_10821_: bool;
    var phi_10836_: bool;
    var phi_10851_: bool;
    var phi_10866_: bool;
    var phi_10881_: bool;
    var phi_10896_: bool;
    var phi_10911_: bool;
    var phi_10926_: bool;
    var phi_10941_: bool;
    var phi_10956_: bool;
    var phi_10971_: bool;
    var phi_10986_: bool;
    var phi_11001_: bool;
    var phi_11016_: bool;
    var phi_11031_: bool;
    var phi_11046_: bool;
    var phi_11061_: bool;
    var phi_11076_: bool;
    var phi_11091_: bool;
    var phi_11106_: bool;
    var phi_11121_: bool;
    var phi_11136_: bool;
    var phi_11151_: bool;
    var phi_11166_: bool;
    var phi_11181_: bool;
    var phi_11196_: bool;
    var phi_11211_: bool;
    var phi_11226_: bool;
    var phi_11241_: bool;
    var phi_11256_: bool;
    var phi_11271_: bool;
    var phi_11286_: bool;
    var phi_11301_: bool;
    var phi_11316_: bool;
    var phi_11331_: bool;
    var phi_11346_: bool;
    var phi_11361_: bool;
    var phi_11376_: bool;
    var phi_11391_: bool;
    var phi_11406_: bool;
    var phi_11421_: bool;
    var phi_11436_: bool;
    var phi_11451_: bool;
    var phi_11466_: bool;
    var phi_11481_: bool;
    var phi_11496_: bool;
    var phi_11511_: bool;
    var phi_11526_: bool;
    var phi_11541_: bool;
    var phi_11556_: bool;
    var phi_11571_: bool;
    var phi_11586_: bool;
    var phi_11601_: bool;
    var phi_7233_: bool;
    var phi_7235_: bool;
    var phi_7236_: bool;
    var phi_7237_: bool;
    var phi_7238_: bool;
    var phi_11616_: bool;
    var phi_11631_: bool;
    var phi_11646_: bool;
    var phi_11661_: bool;
    var phi_7616_: type_27_;
    var phi_7625_: f32;
    var phi_7626_: f32;
    var phi_12423_: bool;
    var phi_308_: f32;
    var phi_310_: f32;
    var phi_12422_: bool;
    var phi_7640_: u32;
    var phi_7663_: bool;
    var phi_7679_: bool;
    var phi_7687_: vec2<f32>;
    var local: f32;
    var phi_7693_: f32;
    var local_1_: f32;
    var phi_7699_: f32;
    var phi_7716_: bool;
    var phi_7722_: f32;
    var phi_7743_: bool;
    var phi_7749_: f32;
    var phi_7760_: bool;
    var phi_7776_: bool;
    var phi_7814_: u32;
    var phi_7815_: u32;
    var phi_12830_: bool;
    var phi_7875_: bool;
    var phi_7893_: bool;
    var phi_7933_: f32;
    var phi_7934_: bool;
    var phi_11676_: bool;
    var phi_8025_: bool;
    var phi_11736_: bool;
    var phi_7990_: bool;
    var phi_8031_: f32;
    var phi_8032_: f32;
    var phi_8033_: f32;
    var phi_8034_: bool;
    var phi_8050_: bool;
    var phi_8070_: u32;

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
            phi_11982_ = false;
            phi_304_ = type_15_(0u, _e139_);
            phi_307_ = 0f;
            phi_309_ = 0f;
            loop {
                let _e199_ = phi_11982_;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_8106_ = true;
                    } else {
                        phi_8106_ = (0.000001f >= _e299_);
                    }
                    let _e303_ = phi_8106_;
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
                                    phi_8226_ = true;
                                } else {
                                    phi_8226_ = (0.000001f >= _e388_);
                                }
                                let _e392_ = phi_8226_;
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
                                    phi_8166_ = true;
                                } else {
                                    phi_8166_ = (0.000001f >= _e352_);
                                }
                                let _e361_ = phi_8166_;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_8286_ = true;
                    } else {
                        phi_8286_ = (0.00001f >= _e320_);
                    }
                    let _e574_ = phi_8286_;
                    let _e575_ = select(_e320_, 0.00001f, _e574_);
                    if (_e426_ != _e426_) {
                        phi_8346_ = true;
                    } else {
                        phi_8346_ = (0f >= _e426_);
                    }
                    let _e579_ = phi_8346_;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
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
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e865_.member;
                    let _e872_ = (256u + _e867_);
                    let _e873_ = (_e872_ < 1536u);
                    if _e873_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e865_.member_1_;
                    let _e876_ = (512u + _e867_);
                    let _e877_ = (_e876_ < 1536u);
                    if _e877_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e876_] = _e865_.member_2_;
                    let _e880_ = (768u + _e867_);
                    let _e881_ = (_e880_ < 1536u);
                    if _e881_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e880_] = _e865_.member_3_;
                    let _e884_ = (1024u + _e867_);
                    let _e885_ = (_e884_ < 1536u);
                    if _e885_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e884_] = _e865_.member_4_;
                    let _e888_ = (1280u + _e867_);
                    let _e889_ = (_e888_ < 1536u);
                    if _e889_ {
                    } else {
                        phi_12422_ = true;
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
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e905_ = scan_scratch[_e902_];
                                let _e906_ = (256u + _e902_);
                                if (_e906_ < 1536u) {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e909_ = scan_scratch[_e906_];
                                let _e910_ = (512u + _e902_);
                                if (_e910_ < 1536u) {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e913_ = scan_scratch[_e910_];
                                let _e914_ = (768u + _e902_);
                                if (_e914_ < 1536u) {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e917_ = scan_scratch[_e914_];
                                let _e918_ = (1024u + _e902_);
                                if (_e918_ < 1536u) {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e921_ = scan_scratch[_e918_];
                                let _e922_ = (1280u + _e902_);
                                if (_e922_ < 1536u) {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e925_ = scan_scratch[_e922_];
                                let _e927_ = select(255u, _e898_, (_e898_ < 255u));
                                let _e928_ = (_e927_ < 1536u);
                                if _e928_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e930_ = scan_scratch[_e927_];
                                let _e931_ = (256u + _e927_);
                                let _e932_ = (_e931_ < 1536u);
                                if _e932_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e934_ = scan_scratch[_e931_];
                                let _e935_ = (512u + _e927_);
                                let _e936_ = (_e935_ < 1536u);
                                if _e936_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e938_ = scan_scratch[_e935_];
                                let _e939_ = (768u + _e927_);
                                let _e940_ = (_e939_ < 1536u);
                                if _e940_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e942_ = scan_scratch[_e939_];
                                let _e943_ = (1024u + _e927_);
                                let _e944_ = (_e943_ < 1536u);
                                if _e944_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e946_ = scan_scratch[_e943_];
                                let _e947_ = (1280u + _e927_);
                                let _e948_ = (_e947_ < 1536u);
                                if _e948_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                let _e950_ = scan_scratch[_e947_];
                                if _e928_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                scan_scratch[_e927_] = ((_e930_ * _e905_) + (_e934_ * _e913_));
                                if _e932_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                scan_scratch[_e931_] = ((_e930_ * _e909_) + (_e934_ * _e917_));
                                if _e936_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                scan_scratch[_e935_] = ((_e938_ * _e905_) + (_e942_ * _e913_));
                                if _e940_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                scan_scratch[_e939_] = ((_e938_ * _e909_) + (_e942_ * _e917_));
                                if _e944_ {
                                } else {
                                    phi_11786_ = true;
                                    break;
                                }
                                scan_scratch[_e943_] = (((_e930_ * _e921_) + (_e934_ * _e925_)) + _e946_);
                                if _e948_ {
                                } else {
                                    phi_11786_ = true;
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
                        phi_11786_ = _e199_;
                        if !(_e893_) {
                            break;
                        }
                    }
                    let _e975_ = phi_11786_;
                    phi_12422_ = _e975_;
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
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e997_ = scan_scratch[_e994_];
                                let _e998_ = (256u + _e994_);
                                let _e999_ = (_e998_ < 1536u);
                                if _e999_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1001_ = scan_scratch[_e998_];
                                let _e1002_ = (512u + _e994_);
                                let _e1003_ = (_e1002_ < 1536u);
                                if _e1003_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1005_ = scan_scratch[_e1002_];
                                let _e1006_ = (768u + _e994_);
                                let _e1007_ = (_e1006_ < 1536u);
                                if _e1007_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1009_ = scan_scratch[_e1006_];
                                let _e1010_ = (1024u + _e994_);
                                let _e1011_ = (_e1010_ < 1536u);
                                if _e1011_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1013_ = scan_scratch[_e1010_];
                                let _e1014_ = (1280u + _e994_);
                                let _e1015_ = (_e1014_ < 1536u);
                                if _e1015_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1017_ = scan_scratch[_e1014_];
                                let _e1019_ = select(255u, _e990_, (_e990_ < 255u));
                                let _e1020_ = (_e1019_ < 1536u);
                                if _e1020_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1022_ = scan_scratch[_e1019_];
                                let _e1023_ = (256u + _e1019_);
                                let _e1024_ = (_e1023_ < 1536u);
                                if _e1024_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1026_ = scan_scratch[_e1023_];
                                let _e1027_ = (512u + _e1019_);
                                let _e1028_ = (_e1027_ < 1536u);
                                if _e1028_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1030_ = scan_scratch[_e1027_];
                                let _e1031_ = (768u + _e1019_);
                                let _e1032_ = (_e1031_ < 1536u);
                                if _e1032_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1034_ = scan_scratch[_e1031_];
                                let _e1035_ = (1024u + _e1019_);
                                let _e1036_ = (_e1035_ < 1536u);
                                if _e1036_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1038_ = scan_scratch[_e1035_];
                                let _e1039_ = (1280u + _e1019_);
                                let _e1040_ = (_e1039_ < 1536u);
                                if _e1040_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                let _e1042_ = scan_scratch[_e1039_];
                                if _e995_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e994_] = _e1022_;
                                if _e999_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e998_] = _e1026_;
                                if _e1003_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1002_] = _e1030_;
                                if _e1007_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1006_] = _e1034_;
                                if _e1011_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1010_] = _e1038_;
                                if _e1015_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1014_] = _e1042_;
                                if _e1020_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1019_] = ((_e997_ * _e1022_) + (_e1001_ * _e1030_));
                                if _e1024_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1023_] = ((_e997_ * _e1026_) + (_e1001_ * _e1034_));
                                if _e1028_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1027_] = ((_e1005_ * _e1022_) + (_e1009_ * _e1030_));
                                if _e1032_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1031_] = ((_e1005_ * _e1026_) + (_e1009_ * _e1034_));
                                if _e1036_ {
                                } else {
                                    phi_12087_ = true;
                                    break;
                                }
                                scan_scratch[_e1035_] = (((_e997_ * _e1038_) + (_e1001_ * _e1042_)) + _e1013_);
                                if _e1040_ {
                                } else {
                                    phi_12087_ = true;
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
                        phi_12087_ = _e975_;
                        if !(_e985_) {
                            break;
                        }
                    }
                    let _e1069_ = phi_12087_;
                    phi_12422_ = _e1069_;
                    if _e1069_ {
                        break;
                    }
                    if _e869_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e873_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e877_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e881_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e885_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    let _e1070_ = scan_scratch[_e884_];
                    if _e889_ {
                    } else {
                        phi_12422_ = true;
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
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e1122_.member;
                    if _e873_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e1122_.member_1_;
                    if _e877_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e876_] = _e1122_.member_2_;
                    if _e881_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e880_] = _e1122_.member_3_;
                    if _e885_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e884_] = _e1122_.member_4_;
                    if _e889_ {
                    } else {
                        phi_12422_ = true;
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
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1143_ = scan_scratch[_e1140_];
                                let _e1144_ = (256u + _e1140_);
                                if (_e1144_ < 1536u) {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1147_ = scan_scratch[_e1144_];
                                let _e1148_ = (512u + _e1140_);
                                if (_e1148_ < 1536u) {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1151_ = scan_scratch[_e1148_];
                                let _e1152_ = (768u + _e1140_);
                                if (_e1152_ < 1536u) {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1155_ = scan_scratch[_e1152_];
                                let _e1156_ = (1024u + _e1140_);
                                if (_e1156_ < 1536u) {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1159_ = scan_scratch[_e1156_];
                                let _e1160_ = (1280u + _e1140_);
                                if (_e1160_ < 1536u) {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1163_ = scan_scratch[_e1160_];
                                let _e1165_ = select(255u, _e1136_, (_e1136_ < 255u));
                                let _e1166_ = (_e1165_ < 1536u);
                                if _e1166_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1168_ = scan_scratch[_e1165_];
                                let _e1169_ = (256u + _e1165_);
                                let _e1170_ = (_e1169_ < 1536u);
                                if _e1170_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1172_ = scan_scratch[_e1169_];
                                let _e1173_ = (512u + _e1165_);
                                let _e1174_ = (_e1173_ < 1536u);
                                if _e1174_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1176_ = scan_scratch[_e1173_];
                                let _e1177_ = (768u + _e1165_);
                                let _e1178_ = (_e1177_ < 1536u);
                                if _e1178_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1180_ = scan_scratch[_e1177_];
                                let _e1181_ = (1024u + _e1165_);
                                let _e1182_ = (_e1181_ < 1536u);
                                if _e1182_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1184_ = scan_scratch[_e1181_];
                                let _e1185_ = (1280u + _e1165_);
                                let _e1186_ = (_e1185_ < 1536u);
                                if _e1186_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                let _e1188_ = scan_scratch[_e1185_];
                                if _e1166_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                scan_scratch[_e1165_] = ((_e1168_ * _e1143_) + (_e1172_ * _e1151_));
                                if _e1170_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                scan_scratch[_e1169_] = ((_e1168_ * _e1147_) + (_e1172_ * _e1155_));
                                if _e1174_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                scan_scratch[_e1173_] = ((_e1176_ * _e1143_) + (_e1180_ * _e1151_));
                                if _e1178_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                scan_scratch[_e1177_] = ((_e1176_ * _e1147_) + (_e1180_ * _e1155_));
                                if _e1182_ {
                                } else {
                                    phi_12102_ = true;
                                    break;
                                }
                                scan_scratch[_e1181_] = (((_e1168_ * _e1159_) + (_e1172_ * _e1163_)) + _e1184_);
                                if _e1186_ {
                                } else {
                                    phi_12102_ = true;
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
                        phi_12102_ = _e1069_;
                        if !(_e1131_) {
                            break;
                        }
                    }
                    let _e1213_ = phi_12102_;
                    phi_12422_ = _e1213_;
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
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1234_ = scan_scratch[_e1231_];
                                let _e1235_ = (256u + _e1231_);
                                let _e1236_ = (_e1235_ < 1536u);
                                if _e1236_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1238_ = scan_scratch[_e1235_];
                                let _e1239_ = (512u + _e1231_);
                                let _e1240_ = (_e1239_ < 1536u);
                                if _e1240_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1242_ = scan_scratch[_e1239_];
                                let _e1243_ = (768u + _e1231_);
                                let _e1244_ = (_e1243_ < 1536u);
                                if _e1244_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1246_ = scan_scratch[_e1243_];
                                let _e1247_ = (1024u + _e1231_);
                                let _e1248_ = (_e1247_ < 1536u);
                                if _e1248_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1250_ = scan_scratch[_e1247_];
                                let _e1251_ = (1280u + _e1231_);
                                let _e1252_ = (_e1251_ < 1536u);
                                if _e1252_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1254_ = scan_scratch[_e1251_];
                                let _e1256_ = select(255u, _e1227_, (_e1227_ < 255u));
                                let _e1257_ = (_e1256_ < 1536u);
                                if _e1257_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1259_ = scan_scratch[_e1256_];
                                let _e1260_ = (256u + _e1256_);
                                let _e1261_ = (_e1260_ < 1536u);
                                if _e1261_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1263_ = scan_scratch[_e1260_];
                                let _e1264_ = (512u + _e1256_);
                                let _e1265_ = (_e1264_ < 1536u);
                                if _e1265_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1267_ = scan_scratch[_e1264_];
                                let _e1268_ = (768u + _e1256_);
                                let _e1269_ = (_e1268_ < 1536u);
                                if _e1269_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1271_ = scan_scratch[_e1268_];
                                let _e1272_ = (1024u + _e1256_);
                                let _e1273_ = (_e1272_ < 1536u);
                                if _e1273_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1275_ = scan_scratch[_e1272_];
                                let _e1276_ = (1280u + _e1256_);
                                let _e1277_ = (_e1276_ < 1536u);
                                if _e1277_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                let _e1279_ = scan_scratch[_e1276_];
                                if _e1232_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1231_] = _e1259_;
                                if _e1236_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1235_] = _e1263_;
                                if _e1240_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1239_] = _e1267_;
                                if _e1244_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1243_] = _e1271_;
                                if _e1248_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1247_] = _e1275_;
                                if _e1252_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1251_] = _e1279_;
                                if _e1257_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1256_] = ((_e1234_ * _e1259_) + (_e1238_ * _e1267_));
                                if _e1261_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1260_] = ((_e1234_ * _e1263_) + (_e1238_ * _e1271_));
                                if _e1265_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1264_] = ((_e1242_ * _e1259_) + (_e1246_ * _e1267_));
                                if _e1269_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1268_] = ((_e1242_ * _e1263_) + (_e1246_ * _e1271_));
                                if _e1273_ {
                                } else {
                                    phi_12122_ = true;
                                    break;
                                }
                                scan_scratch[_e1272_] = (((_e1234_ * _e1275_) + (_e1238_ * _e1279_)) + _e1250_);
                                if _e1277_ {
                                } else {
                                    phi_12122_ = true;
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
                        phi_12122_ = _e1213_;
                        if !(_e1222_) {
                            break;
                        }
                    }
                    let _e1306_ = phi_12122_;
                    phi_12422_ = _e1306_;
                    if _e1306_ {
                        break;
                    }
                    if _e869_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e873_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e877_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e881_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    if _e885_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    let _e1307_ = scan_scratch[_e884_];
                    if _e889_ {
                    } else {
                        phi_12422_ = true;
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
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e1108_.member;
                    if _e873_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e1345_.member;
                    if _e877_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e876_] = _e1108_.member_1_;
                    if _e881_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e880_] = _e1345_.member_1_;
                    if _e885_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e884_] = 0f;
                    if _e889_ {
                    } else {
                        phi_12422_ = true;
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
                                    phi_12137_ = true;
                                    break;
                                }
                                let _e1377_ = scan_scratch[_e1370_.member_1_];
                                let _e1378_ = (256u + _e1370_.member_1_);
                                if (_e1378_ < 1536u) {
                                } else {
                                    phi_12137_ = true;
                                    break;
                                }
                                let _e1381_ = scan_scratch[_e1378_];
                                let _e1382_ = (512u + _e1370_.member_1_);
                                if (_e1382_ < 1536u) {
                                } else {
                                    phi_12137_ = true;
                                    break;
                                }
                                let _e1385_ = scan_scratch[_e1382_];
                                let _e1386_ = (768u + _e1370_.member_1_);
                                if (_e1386_ < 1536u) {
                                } else {
                                    phi_12137_ = true;
                                    break;
                                }
                                let _e1389_ = scan_scratch[_e1386_];
                                let _e1390_ = (_e736_ * _e1377_);
                                let _e1391_ = (_e736_ * _e1381_);
                                let _e1394_ = bitcast<u32>((_e625_ * _e1385_));
                                let _e1395_ = (_e1394_ & 2139095040u);
                                let _e1396_ = (_e1395_ == 2139095040u);
                                if _e1396_ {
                                    phi_4193_ = 0u;
                                } else {
                                    if (_e1395_ == 0u) {
                                        phi_4192_ = (_e1394_ & 2147483648u);
                                    } else {
                                        phi_4192_ = _e1394_;
                                    }
                                    let _e1400_ = phi_4192_;
                                    phi_4193_ = _e1400_;
                                }
                                let _e1402_ = phi_4193_;
                                if _e1396_ {
                                    phi_4207_ = 0u;
                                } else {
                                    if (_e1395_ == 0u) {
                                        phi_4206_ = (_e1394_ & 2147483648u);
                                    } else {
                                        phi_4206_ = _e1394_;
                                    }
                                    let _e1407_ = phi_4206_;
                                    phi_4207_ = _e1407_;
                                }
                                let _e1409_ = phi_4207_;
                                if _e1396_ {
                                    phi_4221_ = 0u;
                                } else {
                                    if (_e1395_ == 0u) {
                                        phi_4220_ = (_e1394_ & 2147483648u);
                                    } else {
                                        phi_4220_ = _e1394_;
                                    }
                                    let _e1414_ = phi_4220_;
                                    phi_4221_ = _e1414_;
                                }
                                let _e1416_ = phi_4221_;
                                if _e1396_ {
                                    phi_4235_ = 0u;
                                } else {
                                    if (_e1395_ == 0u) {
                                        phi_4234_ = (_e1394_ & 2147483648u);
                                    } else {
                                        phi_4234_ = _e1394_;
                                    }
                                    let _e1421_ = phi_4234_;
                                    phi_4235_ = _e1421_;
                                }
                                let _e1423_ = phi_4235_;
                                let _e1441_ = (_e558_ <= 0f);
                                let _e1446_ = (1f - select(select(select(0f, 1f, (1f <= _e462_)), _e462_, (_e462_ <= 1f)), 0f, (_e462_ <= 0f)));
                                let _e1447_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1402_)) & 2147483647u))), 0f, _e1441_));
                                let _e1448_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1409_)) & 2147483647u))), 0f, _e1441_));
                                let _e1449_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1416_)) & 2147483647u))), 0f, _e1441_));
                                let _e1450_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1423_)) & 2147483647u))), 0f, _e1441_));
                                if (_e1447_ != _e1447_) {
                                    phi_8406_ = true;
                                } else {
                                    phi_8406_ = (0f >= _e1447_);
                                }
                                let _e1454_ = phi_8406_;
                                let _e1455_ = select(_e1447_, 0f, _e1454_);
                                if (_e1448_ != _e1448_) {
                                    phi_8421_ = true;
                                } else {
                                    phi_8421_ = (0f >= _e1448_);
                                }
                                let _e1459_ = phi_8421_;
                                let _e1460_ = select(_e1448_, 0f, _e1459_);
                                if (_e1449_ != _e1449_) {
                                    phi_8436_ = true;
                                } else {
                                    phi_8436_ = (0f >= _e1449_);
                                }
                                let _e1464_ = phi_8436_;
                                let _e1465_ = select(_e1449_, 0f, _e1464_);
                                if (_e1450_ != _e1450_) {
                                    phi_8451_ = true;
                                } else {
                                    phi_8451_ = (0f >= _e1450_);
                                }
                                let _e1469_ = phi_8451_;
                                let _e1470_ = select(_e1450_, 0f, _e1469_);
                                if (_e1455_ != _e1455_) {
                                    phi_8466_ = true;
                                } else {
                                    phi_8466_ = (1f <= _e1455_);
                                }
                                let _e1474_ = phi_8466_;
                                let _e1475_ = select(_e1455_, 1f, _e1474_);
                                if (_e1460_ != _e1460_) {
                                    phi_8481_ = true;
                                } else {
                                    phi_8481_ = (1f <= _e1460_);
                                }
                                let _e1479_ = phi_8481_;
                                let _e1480_ = select(_e1460_, 1f, _e1479_);
                                if (_e1465_ != _e1465_) {
                                    phi_8496_ = true;
                                } else {
                                    phi_8496_ = (1f <= _e1465_);
                                }
                                let _e1484_ = phi_8496_;
                                let _e1485_ = select(_e1465_, 1f, _e1484_);
                                if (_e1470_ != _e1470_) {
                                    phi_8511_ = true;
                                } else {
                                    phi_8511_ = (1f <= _e1470_);
                                }
                                let _e1489_ = phi_8511_;
                                let _e1490_ = select(_e1470_, 1f, _e1489_);
                                let _e1493_ = (_e540_ * _e1355_.member[0]);
                                let _e1496_ = (_e540_ * _e1355_.member[1]);
                                let _e1499_ = (_e540_ * _e1355_.member[2]);
                                let _e1502_ = (_e540_ * _e1355_.member[3]);
                                let _e1503_ = (1f - _e564_);
                                let _e1504_ = (select(select(select(0f, 1f, (1f <= _e480_)), _e480_, (_e480_ <= 1f)), 0f, (_e480_ <= 0f)) * _e1503_);
                                let _e1506_ = (_e1390_ - (_e1504_ * _e1493_));
                                let _e1508_ = (_e1390_ - (_e1504_ * _e1496_));
                                let _e1510_ = (_e1390_ - (_e1504_ * _e1499_));
                                let _e1512_ = (_e1390_ - (_e1504_ * _e1502_));
                                if (_e1506_ != _e1506_) {
                                    phi_8526_ = true;
                                } else {
                                    phi_8526_ = (-1f >= _e1506_);
                                }
                                let _e1516_ = phi_8526_;
                                let _e1517_ = select(_e1506_, -1f, _e1516_);
                                if (_e1508_ != _e1508_) {
                                    phi_8541_ = true;
                                } else {
                                    phi_8541_ = (-1f >= _e1508_);
                                }
                                let _e1521_ = phi_8541_;
                                let _e1522_ = select(_e1508_, -1f, _e1521_);
                                if (_e1510_ != _e1510_) {
                                    phi_8556_ = true;
                                } else {
                                    phi_8556_ = (-1f >= _e1510_);
                                }
                                let _e1526_ = phi_8556_;
                                let _e1527_ = select(_e1510_, -1f, _e1526_);
                                if (_e1512_ != _e1512_) {
                                    phi_8571_ = true;
                                } else {
                                    phi_8571_ = (-1f >= _e1512_);
                                }
                                let _e1531_ = phi_8571_;
                                let _e1532_ = select(_e1512_, -1f, _e1531_);
                                if (_e1517_ != _e1517_) {
                                    phi_8586_ = true;
                                } else {
                                    phi_8586_ = (1f <= _e1517_);
                                }
                                let _e1536_ = phi_8586_;
                                let _e1537_ = select(_e1517_, 1f, _e1536_);
                                if (_e1522_ != _e1522_) {
                                    phi_8601_ = true;
                                } else {
                                    phi_8601_ = (1f <= _e1522_);
                                }
                                let _e1541_ = phi_8601_;
                                let _e1542_ = select(_e1522_, 1f, _e1541_);
                                if (_e1527_ != _e1527_) {
                                    phi_8616_ = true;
                                } else {
                                    phi_8616_ = (1f <= _e1527_);
                                }
                                let _e1546_ = phi_8616_;
                                let _e1547_ = select(_e1527_, 1f, _e1546_);
                                if (_e1532_ != _e1532_) {
                                    phi_8631_ = true;
                                } else {
                                    phi_8631_ = (1f <= _e1532_);
                                }
                                let _e1551_ = phi_8631_;
                                let _e1552_ = select(_e1532_, 1f, _e1551_);
                                let _e1571_ = bitcast<f32>((bitcast<u32>(_e1506_) & 2147483647u));
                                let _e1574_ = bitcast<f32>((bitcast<u32>(_e1508_) & 2147483647u));
                                let _e1577_ = bitcast<f32>((bitcast<u32>(_e1510_) & 2147483647u));
                                let _e1580_ = bitcast<f32>((bitcast<u32>(_e1512_) & 2147483647u));
                                if (_e1571_ != _e1571_) {
                                    phi_8646_ = true;
                                } else {
                                    phi_8646_ = (1000000000000000f <= _e1571_);
                                }
                                let _e1584_ = phi_8646_;
                                let _e1585_ = select(_e1571_, 1000000000000000f, _e1584_);
                                if (_e1574_ != _e1574_) {
                                    phi_8661_ = true;
                                } else {
                                    phi_8661_ = (1000000000000000f <= _e1574_);
                                }
                                let _e1589_ = phi_8661_;
                                let _e1590_ = select(_e1574_, 1000000000000000f, _e1589_);
                                if (_e1577_ != _e1577_) {
                                    phi_8676_ = true;
                                } else {
                                    phi_8676_ = (1000000000000000f <= _e1577_);
                                }
                                let _e1594_ = phi_8676_;
                                let _e1595_ = select(_e1577_, 1000000000000000f, _e1594_);
                                if (_e1580_ != _e1580_) {
                                    phi_8691_ = true;
                                } else {
                                    phi_8691_ = (1000000000000000f <= _e1580_);
                                }
                                let _e1599_ = phi_8691_;
                                let _e1600_ = select(_e1580_, 1000000000000000f, _e1599_);
                                let _e1601_ = (_e1585_ * _e1585_);
                                let _e1602_ = (_e1590_ * _e1590_);
                                let _e1603_ = (_e1595_ * _e1595_);
                                let _e1604_ = (_e1600_ * _e1600_);
                                let _e1605_ = (_e1601_ + 0.0000010000001f);
                                let _e1606_ = (_e1602_ + 0.0000010000001f);
                                let _e1607_ = (_e1603_ + 0.0000010000001f);
                                let _e1608_ = (_e1604_ + 0.0000010000001f);
                                if (_e1605_ != _e1605_) {
                                    phi_8706_ = true;
                                } else {
                                    phi_8706_ = (0.000000000001f >= _e1605_);
                                }
                                let _e1612_ = phi_8706_;
                                let _e1613_ = select(_e1605_, 0.000000000001f, _e1612_);
                                if (_e1606_ != _e1606_) {
                                    phi_8721_ = true;
                                } else {
                                    phi_8721_ = (0.000000000001f >= _e1606_);
                                }
                                let _e1617_ = phi_8721_;
                                let _e1618_ = select(_e1606_, 0.000000000001f, _e1617_);
                                if (_e1607_ != _e1607_) {
                                    phi_8736_ = true;
                                } else {
                                    phi_8736_ = (0.000000000001f >= _e1607_);
                                }
                                let _e1622_ = phi_8736_;
                                let _e1623_ = select(_e1607_, 0.000000000001f, _e1622_);
                                if (_e1608_ != _e1608_) {
                                    phi_8751_ = true;
                                } else {
                                    phi_8751_ = (0.000000000001f >= _e1608_);
                                }
                                let _e1627_ = phi_8751_;
                                let _e1628_ = select(_e1608_, 0.000000000001f, _e1627_);
                                let _e1629_ = (_e1613_ * 0.5f);
                                let _e1630_ = (_e1618_ * 0.5f);
                                let _e1631_ = (_e1623_ * 0.5f);
                                let _e1632_ = (_e1628_ * 0.5f);
                                let _e1637_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1613_) >> bitcast<u32>(1i))));
                                let _e1642_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1618_) >> bitcast<u32>(1i))));
                                let _e1647_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1623_) >> bitcast<u32>(1i))));
                                let _e1652_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1628_) >> bitcast<u32>(1i))));
                                let _e1665_ = (_e1637_ * (1.5f - (_e1629_ * (_e1637_ * _e1637_))));
                                let _e1666_ = (_e1642_ * (1.5f - (_e1630_ * (_e1642_ * _e1642_))));
                                let _e1667_ = (_e1647_ * (1.5f - (_e1631_ * (_e1647_ * _e1647_))));
                                let _e1668_ = (_e1652_ * (1.5f - (_e1632_ * (_e1652_ * _e1652_))));
                                let _e1681_ = (_e1665_ * (1.5f - (_e1629_ * (_e1665_ * _e1665_))));
                                let _e1682_ = (_e1666_ * (1.5f - (_e1630_ * (_e1666_ * _e1666_))));
                                let _e1683_ = (_e1667_ * (1.5f - (_e1631_ * (_e1667_ * _e1667_))));
                                let _e1684_ = (_e1668_ * (1.5f - (_e1632_ * (_e1668_ * _e1668_))));
                                let _e1701_ = (_e1605_ * (_e1681_ * (1.5f - (_e1629_ * (_e1681_ * _e1681_)))));
                                let _e1702_ = (_e1606_ * (_e1682_ * (1.5f - (_e1630_ * (_e1682_ * _e1682_)))));
                                let _e1703_ = (_e1607_ * (_e1683_ * (1.5f - (_e1631_ * (_e1683_ * _e1683_)))));
                                let _e1704_ = (_e1608_ * (_e1684_ * (1.5f - (_e1632_ * (_e1684_ * _e1684_)))));
                                let _e1705_ = (_e1701_ + 0.001f);
                                let _e1706_ = (_e1702_ + 0.001f);
                                let _e1707_ = (_e1703_ + 0.001f);
                                let _e1708_ = (_e1704_ + 0.001f);
                                if (_e1705_ != _e1705_) {
                                    phi_8766_ = true;
                                } else {
                                    phi_8766_ = (0.000001f >= _e1705_);
                                }
                                let _e1712_ = phi_8766_;
                                let _e1713_ = select(_e1705_, 0.000001f, _e1712_);
                                if (_e1706_ != _e1706_) {
                                    phi_8781_ = true;
                                } else {
                                    phi_8781_ = (0.000001f >= _e1706_);
                                }
                                let _e1717_ = phi_8781_;
                                let _e1718_ = select(_e1706_, 0.000001f, _e1717_);
                                if (_e1707_ != _e1707_) {
                                    phi_8796_ = true;
                                } else {
                                    phi_8796_ = (0.000001f >= _e1707_);
                                }
                                let _e1722_ = phi_8796_;
                                let _e1723_ = select(_e1707_, 0.000001f, _e1722_);
                                if (_e1708_ != _e1708_) {
                                    phi_8811_ = true;
                                } else {
                                    phi_8811_ = (0.000001f >= _e1708_);
                                }
                                let _e1727_ = phi_8811_;
                                let _e1728_ = select(_e1708_, 0.000001f, _e1727_);
                                let _e1731_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1713_)));
                                let _e1734_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1718_)));
                                let _e1737_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1723_)));
                                let _e1740_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1728_)));
                                let _e1750_ = (_e1731_ + (_e1731_ * (1f - (_e1713_ * _e1731_))));
                                let _e1752_ = (_e1734_ + (_e1734_ * (1f - (_e1718_ * _e1734_))));
                                let _e1754_ = (_e1737_ + (_e1737_ * (1f - (_e1723_ * _e1737_))));
                                let _e1756_ = (_e1740_ + (_e1740_ * (1f - (_e1728_ * _e1740_))));
                                let _e1766_ = (_e1750_ + (_e1750_ * (1f - (_e1713_ * _e1750_))));
                                let _e1768_ = (_e1752_ + (_e1752_ * (1f - (_e1718_ * _e1752_))));
                                let _e1770_ = (_e1754_ + (_e1754_ * (1f - (_e1723_ * _e1754_))));
                                let _e1772_ = (_e1756_ + (_e1756_ * (1f - (_e1728_ * _e1756_))));
                                let _e1782_ = (_e1766_ + (_e1766_ * (1f - (_e1713_ * _e1766_))));
                                let _e1784_ = (_e1768_ + (_e1768_ * (1f - (_e1718_ * _e1768_))));
                                let _e1786_ = (_e1770_ + (_e1770_ * (1f - (_e1723_ * _e1770_))));
                                let _e1788_ = (_e1772_ + (_e1772_ * (1f - (_e1728_ * _e1772_))));
                                let _e1789_ = (_e1701_ + _e1585_);
                                let _e1790_ = (_e1702_ + _e1590_);
                                let _e1791_ = (_e1703_ + _e1595_);
                                let _e1792_ = (_e1704_ + _e1600_);
                                if (_e1789_ != _e1789_) {
                                    phi_8826_ = true;
                                } else {
                                    phi_8826_ = (0.000001f >= _e1789_);
                                }
                                let _e1796_ = phi_8826_;
                                let _e1797_ = select(_e1789_, 0.000001f, _e1796_);
                                if (_e1790_ != _e1790_) {
                                    phi_8841_ = true;
                                } else {
                                    phi_8841_ = (0.000001f >= _e1790_);
                                }
                                let _e1801_ = phi_8841_;
                                let _e1802_ = select(_e1790_, 0.000001f, _e1801_);
                                if (_e1791_ != _e1791_) {
                                    phi_8856_ = true;
                                } else {
                                    phi_8856_ = (0.000001f >= _e1791_);
                                }
                                let _e1806_ = phi_8856_;
                                let _e1807_ = select(_e1791_, 0.000001f, _e1806_);
                                if (_e1792_ != _e1792_) {
                                    phi_8871_ = true;
                                } else {
                                    phi_8871_ = (0.000001f >= _e1792_);
                                }
                                let _e1811_ = phi_8871_;
                                let _e1812_ = select(_e1792_, 0.000001f, _e1811_);
                                let _e1815_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1797_)));
                                let _e1818_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1802_)));
                                let _e1821_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1807_)));
                                let _e1824_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1812_)));
                                let _e1834_ = (_e1815_ + (_e1815_ * (1f - (_e1797_ * _e1815_))));
                                let _e1836_ = (_e1818_ + (_e1818_ * (1f - (_e1802_ * _e1818_))));
                                let _e1838_ = (_e1821_ + (_e1821_ * (1f - (_e1807_ * _e1821_))));
                                let _e1840_ = (_e1824_ + (_e1824_ * (1f - (_e1812_ * _e1824_))));
                                let _e1850_ = (_e1834_ + (_e1834_ * (1f - (_e1797_ * _e1834_))));
                                let _e1852_ = (_e1836_ + (_e1836_ * (1f - (_e1802_ * _e1836_))));
                                let _e1854_ = (_e1838_ + (_e1838_ * (1f - (_e1807_ * _e1838_))));
                                let _e1856_ = (_e1840_ + (_e1840_ * (1f - (_e1812_ * _e1840_))));
                                let _e1893_ = (_e1571_ <= 1000000000000000f);
                                let _e1895_ = (_e1574_ <= 1000000000000000f);
                                let _e1897_ = (_e1577_ <= 1000000000000000f);
                                let _e1899_ = (_e1580_ <= 1000000000000000f);
                                let _e1930_ = select(select(select(_e1571_, (0.5f * (_e1585_ + (_e1601_ * _e1782_))), _e1893_), select(-0.0005f, -(((0.0005f * (_e1585_ * (_e1850_ + (_e1850_ * (1f - (_e1797_ * _e1850_)))))) * (1f + (_e1585_ * _e1782_)))), _e1893_), (_e1506_ <= 0f)), 0f, (_e1571_ <= 0f));
                                let _e1932_ = select(select(select(_e1574_, (0.5f * (_e1590_ + (_e1602_ * _e1784_))), _e1895_), select(-0.0005f, -(((0.0005f * (_e1590_ * (_e1852_ + (_e1852_ * (1f - (_e1802_ * _e1852_)))))) * (1f + (_e1590_ * _e1784_)))), _e1895_), (_e1508_ <= 0f)), 0f, (_e1574_ <= 0f));
                                let _e1934_ = select(select(select(_e1577_, (0.5f * (_e1595_ + (_e1603_ * _e1786_))), _e1897_), select(-0.0005f, -(((0.0005f * (_e1595_ * (_e1854_ + (_e1854_ * (1f - (_e1807_ * _e1854_)))))) * (1f + (_e1595_ * _e1786_)))), _e1897_), (_e1510_ <= 0f)), 0f, (_e1577_ <= 0f));
                                let _e1936_ = select(select(select(_e1580_, (0.5f * (_e1600_ + (_e1604_ * _e1788_))), _e1899_), select(-0.0005f, -(((0.0005f * (_e1600_ * (_e1856_ + (_e1856_ * (1f - (_e1812_ * _e1856_)))))) * (1f + (_e1600_ * _e1788_)))), _e1899_), (_e1512_ <= 0f)), 0f, (_e1580_ <= 0f));
                                let _e1949_ = (_e564_ <= 0f);
                                let _e1974_ = bitcast<u32>((((1f - (_e564_ * _e1475_)) * _e1493_) + (_e1475_ * select(((_e1503_ * _e1930_) + (_e564_ * (_e1537_ * (1f - (0.33333334f * (_e1537_ * _e1537_)))))), _e1930_, _e1949_))));
                                let _e1975_ = (_e1974_ & 2139095040u);
                                if (_e1975_ == 2139095040u) {
                                    phi_4941_ = 0u;
                                } else {
                                    if (_e1975_ == 0u) {
                                        phi_4940_ = (_e1974_ & 2147483648u);
                                    } else {
                                        phi_4940_ = _e1974_;
                                    }
                                    let _e1980_ = phi_4940_;
                                    phi_4941_ = _e1980_;
                                }
                                let _e1982_ = phi_4941_;
                                let _e1983_ = bitcast<f32>(_e1982_);
                                let _e1984_ = bitcast<u32>((((1f - (_e564_ * _e1480_)) * _e1496_) + (_e1480_ * select(((_e1503_ * _e1932_) + (_e564_ * (_e1542_ * (1f - (0.33333334f * (_e1542_ * _e1542_)))))), _e1932_, _e1949_))));
                                let _e1985_ = (_e1984_ & 2139095040u);
                                if (_e1985_ == 2139095040u) {
                                    phi_4955_ = 0u;
                                } else {
                                    if (_e1985_ == 0u) {
                                        phi_4954_ = (_e1984_ & 2147483648u);
                                    } else {
                                        phi_4954_ = _e1984_;
                                    }
                                    let _e1990_ = phi_4954_;
                                    phi_4955_ = _e1990_;
                                }
                                let _e1992_ = phi_4955_;
                                let _e1994_ = bitcast<u32>((((1f - (_e564_ * _e1485_)) * _e1499_) + (_e1485_ * select(((_e1503_ * _e1934_) + (_e564_ * (_e1547_ * (1f - (0.33333334f * (_e1547_ * _e1547_)))))), _e1934_, _e1949_))));
                                let _e1995_ = (_e1994_ & 2139095040u);
                                if (_e1995_ == 2139095040u) {
                                    phi_4969_ = 0u;
                                } else {
                                    if (_e1995_ == 0u) {
                                        phi_4968_ = (_e1994_ & 2147483648u);
                                    } else {
                                        phi_4968_ = _e1994_;
                                    }
                                    let _e2000_ = phi_4968_;
                                    phi_4969_ = _e2000_;
                                }
                                let _e2002_ = phi_4969_;
                                let _e2004_ = bitcast<u32>((((1f - (_e564_ * _e1490_)) * _e1502_) + (_e1490_ * select(((_e1503_ * _e1936_) + (_e564_ * (_e1552_ * (1f - (0.33333334f * (_e1552_ * _e1552_)))))), _e1936_, _e1949_))));
                                let _e2005_ = (_e2004_ & 2139095040u);
                                if (_e2005_ == 2139095040u) {
                                    phi_4983_ = 0u;
                                } else {
                                    if (_e2005_ == 0u) {
                                        phi_4982_ = (_e2004_ & 2147483648u);
                                    } else {
                                        phi_4982_ = _e2004_;
                                    }
                                    let _e2010_ = phi_4982_;
                                    phi_4983_ = _e2010_;
                                }
                                let _e2012_ = phi_4983_;
                                let _e2016_ = bitcast<u32>((_e625_ * _e1389_));
                                let _e2017_ = (_e2016_ & 2139095040u);
                                let _e2018_ = (_e2017_ == 2139095040u);
                                if _e2018_ {
                                    phi_5003_ = 0u;
                                } else {
                                    if (_e2017_ == 0u) {
                                        phi_5002_ = (_e2016_ & 2147483648u);
                                    } else {
                                        phi_5002_ = _e2016_;
                                    }
                                    let _e2022_ = phi_5002_;
                                    phi_5003_ = _e2022_;
                                }
                                let _e2024_ = phi_5003_;
                                if _e2018_ {
                                    phi_5017_ = 0u;
                                } else {
                                    if (_e2017_ == 0u) {
                                        phi_5016_ = (_e2016_ & 2147483648u);
                                    } else {
                                        phi_5016_ = _e2016_;
                                    }
                                    let _e2029_ = phi_5016_;
                                    phi_5017_ = _e2029_;
                                }
                                let _e2031_ = phi_5017_;
                                if _e2018_ {
                                    phi_5031_ = 0u;
                                } else {
                                    if (_e2017_ == 0u) {
                                        phi_5030_ = (_e2016_ & 2147483648u);
                                    } else {
                                        phi_5030_ = _e2016_;
                                    }
                                    let _e2036_ = phi_5030_;
                                    phi_5031_ = _e2036_;
                                }
                                let _e2038_ = phi_5031_;
                                if _e2018_ {
                                    phi_5045_ = 0u;
                                } else {
                                    if (_e2017_ == 0u) {
                                        phi_5044_ = (_e2016_ & 2147483648u);
                                    } else {
                                        phi_5044_ = _e2016_;
                                    }
                                    let _e2043_ = phi_5044_;
                                    phi_5045_ = _e2043_;
                                }
                                let _e2045_ = phi_5045_;
                                let _e2067_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2024_)) & 2147483647u))), 0f, _e1441_));
                                let _e2068_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2031_)) & 2147483647u))), 0f, _e1441_));
                                let _e2069_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2038_)) & 2147483647u))), 0f, _e1441_));
                                let _e2070_ = (_e1446_ + select((_e558_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2045_)) & 2147483647u))), 0f, _e1441_));
                                if (_e2067_ != _e2067_) {
                                    phi_9006_ = true;
                                } else {
                                    phi_9006_ = (0f >= _e2067_);
                                }
                                let _e2074_ = phi_9006_;
                                let _e2075_ = select(_e2067_, 0f, _e2074_);
                                if (_e2068_ != _e2068_) {
                                    phi_9021_ = true;
                                } else {
                                    phi_9021_ = (0f >= _e2068_);
                                }
                                let _e2079_ = phi_9021_;
                                let _e2080_ = select(_e2068_, 0f, _e2079_);
                                if (_e2069_ != _e2069_) {
                                    phi_9036_ = true;
                                } else {
                                    phi_9036_ = (0f >= _e2069_);
                                }
                                let _e2084_ = phi_9036_;
                                let _e2085_ = select(_e2069_, 0f, _e2084_);
                                if (_e2070_ != _e2070_) {
                                    phi_9051_ = true;
                                } else {
                                    phi_9051_ = (0f >= _e2070_);
                                }
                                let _e2089_ = phi_9051_;
                                let _e2090_ = select(_e2070_, 0f, _e2089_);
                                if (_e2075_ != _e2075_) {
                                    phi_9066_ = true;
                                } else {
                                    phi_9066_ = (1f <= _e2075_);
                                }
                                let _e2094_ = phi_9066_;
                                let _e2095_ = select(_e2075_, 1f, _e2094_);
                                if (_e2080_ != _e2080_) {
                                    phi_9081_ = true;
                                } else {
                                    phi_9081_ = (1f <= _e2080_);
                                }
                                let _e2099_ = phi_9081_;
                                let _e2100_ = select(_e2080_, 1f, _e2099_);
                                if (_e2085_ != _e2085_) {
                                    phi_9096_ = true;
                                } else {
                                    phi_9096_ = (1f <= _e2085_);
                                }
                                let _e2104_ = phi_9096_;
                                let _e2105_ = select(_e2085_, 1f, _e2104_);
                                if (_e2090_ != _e2090_) {
                                    phi_9111_ = true;
                                } else {
                                    phi_9111_ = (1f <= _e2090_);
                                }
                                let _e2109_ = phi_9111_;
                                let _e2110_ = select(_e2090_, 1f, _e2109_);
                                let _e2113_ = (_e540_ * _e1353_.member[0]);
                                let _e2116_ = (_e540_ * _e1353_.member[1]);
                                let _e2119_ = (_e540_ * _e1353_.member[2]);
                                let _e2122_ = (_e540_ * _e1353_.member[3]);
                                let _e2124_ = (_e1391_ - (_e1504_ * _e2113_));
                                let _e2126_ = (_e1391_ - (_e1504_ * _e2116_));
                                let _e2128_ = (_e1391_ - (_e1504_ * _e2119_));
                                let _e2130_ = (_e1391_ - (_e1504_ * _e2122_));
                                if (_e2124_ != _e2124_) {
                                    phi_9126_ = true;
                                } else {
                                    phi_9126_ = (-1f >= _e2124_);
                                }
                                let _e2134_ = phi_9126_;
                                let _e2135_ = select(_e2124_, -1f, _e2134_);
                                if (_e2126_ != _e2126_) {
                                    phi_9141_ = true;
                                } else {
                                    phi_9141_ = (-1f >= _e2126_);
                                }
                                let _e2139_ = phi_9141_;
                                let _e2140_ = select(_e2126_, -1f, _e2139_);
                                if (_e2128_ != _e2128_) {
                                    phi_9156_ = true;
                                } else {
                                    phi_9156_ = (-1f >= _e2128_);
                                }
                                let _e2144_ = phi_9156_;
                                let _e2145_ = select(_e2128_, -1f, _e2144_);
                                if (_e2130_ != _e2130_) {
                                    phi_9171_ = true;
                                } else {
                                    phi_9171_ = (-1f >= _e2130_);
                                }
                                let _e2149_ = phi_9171_;
                                let _e2150_ = select(_e2130_, -1f, _e2149_);
                                if (_e2135_ != _e2135_) {
                                    phi_9186_ = true;
                                } else {
                                    phi_9186_ = (1f <= _e2135_);
                                }
                                let _e2154_ = phi_9186_;
                                let _e2155_ = select(_e2135_, 1f, _e2154_);
                                if (_e2140_ != _e2140_) {
                                    phi_9201_ = true;
                                } else {
                                    phi_9201_ = (1f <= _e2140_);
                                }
                                let _e2159_ = phi_9201_;
                                let _e2160_ = select(_e2140_, 1f, _e2159_);
                                if (_e2145_ != _e2145_) {
                                    phi_9216_ = true;
                                } else {
                                    phi_9216_ = (1f <= _e2145_);
                                }
                                let _e2164_ = phi_9216_;
                                let _e2165_ = select(_e2145_, 1f, _e2164_);
                                if (_e2150_ != _e2150_) {
                                    phi_9231_ = true;
                                } else {
                                    phi_9231_ = (1f <= _e2150_);
                                }
                                let _e2169_ = phi_9231_;
                                let _e2170_ = select(_e2150_, 1f, _e2169_);
                                let _e2189_ = bitcast<f32>((bitcast<u32>(_e2124_) & 2147483647u));
                                let _e2192_ = bitcast<f32>((bitcast<u32>(_e2126_) & 2147483647u));
                                let _e2195_ = bitcast<f32>((bitcast<u32>(_e2128_) & 2147483647u));
                                let _e2198_ = bitcast<f32>((bitcast<u32>(_e2130_) & 2147483647u));
                                if (_e2189_ != _e2189_) {
                                    phi_9246_ = true;
                                } else {
                                    phi_9246_ = (1000000000000000f <= _e2189_);
                                }
                                let _e2202_ = phi_9246_;
                                let _e2203_ = select(_e2189_, 1000000000000000f, _e2202_);
                                if (_e2192_ != _e2192_) {
                                    phi_9261_ = true;
                                } else {
                                    phi_9261_ = (1000000000000000f <= _e2192_);
                                }
                                let _e2207_ = phi_9261_;
                                let _e2208_ = select(_e2192_, 1000000000000000f, _e2207_);
                                if (_e2195_ != _e2195_) {
                                    phi_9276_ = true;
                                } else {
                                    phi_9276_ = (1000000000000000f <= _e2195_);
                                }
                                let _e2212_ = phi_9276_;
                                let _e2213_ = select(_e2195_, 1000000000000000f, _e2212_);
                                if (_e2198_ != _e2198_) {
                                    phi_9291_ = true;
                                } else {
                                    phi_9291_ = (1000000000000000f <= _e2198_);
                                }
                                let _e2217_ = phi_9291_;
                                let _e2218_ = select(_e2198_, 1000000000000000f, _e2217_);
                                let _e2219_ = (_e2203_ * _e2203_);
                                let _e2220_ = (_e2208_ * _e2208_);
                                let _e2221_ = (_e2213_ * _e2213_);
                                let _e2222_ = (_e2218_ * _e2218_);
                                let _e2223_ = (_e2219_ + 0.0000010000001f);
                                let _e2224_ = (_e2220_ + 0.0000010000001f);
                                let _e2225_ = (_e2221_ + 0.0000010000001f);
                                let _e2226_ = (_e2222_ + 0.0000010000001f);
                                if (_e2223_ != _e2223_) {
                                    phi_9306_ = true;
                                } else {
                                    phi_9306_ = (0.000000000001f >= _e2223_);
                                }
                                let _e2230_ = phi_9306_;
                                let _e2231_ = select(_e2223_, 0.000000000001f, _e2230_);
                                if (_e2224_ != _e2224_) {
                                    phi_9321_ = true;
                                } else {
                                    phi_9321_ = (0.000000000001f >= _e2224_);
                                }
                                let _e2235_ = phi_9321_;
                                let _e2236_ = select(_e2224_, 0.000000000001f, _e2235_);
                                if (_e2225_ != _e2225_) {
                                    phi_9336_ = true;
                                } else {
                                    phi_9336_ = (0.000000000001f >= _e2225_);
                                }
                                let _e2240_ = phi_9336_;
                                let _e2241_ = select(_e2225_, 0.000000000001f, _e2240_);
                                if (_e2226_ != _e2226_) {
                                    phi_9351_ = true;
                                } else {
                                    phi_9351_ = (0.000000000001f >= _e2226_);
                                }
                                let _e2245_ = phi_9351_;
                                let _e2246_ = select(_e2226_, 0.000000000001f, _e2245_);
                                let _e2247_ = (_e2231_ * 0.5f);
                                let _e2248_ = (_e2236_ * 0.5f);
                                let _e2249_ = (_e2241_ * 0.5f);
                                let _e2250_ = (_e2246_ * 0.5f);
                                let _e2255_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2231_) >> bitcast<u32>(1i))));
                                let _e2260_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2236_) >> bitcast<u32>(1i))));
                                let _e2265_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2241_) >> bitcast<u32>(1i))));
                                let _e2270_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2246_) >> bitcast<u32>(1i))));
                                let _e2283_ = (_e2255_ * (1.5f - (_e2247_ * (_e2255_ * _e2255_))));
                                let _e2284_ = (_e2260_ * (1.5f - (_e2248_ * (_e2260_ * _e2260_))));
                                let _e2285_ = (_e2265_ * (1.5f - (_e2249_ * (_e2265_ * _e2265_))));
                                let _e2286_ = (_e2270_ * (1.5f - (_e2250_ * (_e2270_ * _e2270_))));
                                let _e2299_ = (_e2283_ * (1.5f - (_e2247_ * (_e2283_ * _e2283_))));
                                let _e2300_ = (_e2284_ * (1.5f - (_e2248_ * (_e2284_ * _e2284_))));
                                let _e2301_ = (_e2285_ * (1.5f - (_e2249_ * (_e2285_ * _e2285_))));
                                let _e2302_ = (_e2286_ * (1.5f - (_e2250_ * (_e2286_ * _e2286_))));
                                let _e2319_ = (_e2223_ * (_e2299_ * (1.5f - (_e2247_ * (_e2299_ * _e2299_)))));
                                let _e2320_ = (_e2224_ * (_e2300_ * (1.5f - (_e2248_ * (_e2300_ * _e2300_)))));
                                let _e2321_ = (_e2225_ * (_e2301_ * (1.5f - (_e2249_ * (_e2301_ * _e2301_)))));
                                let _e2322_ = (_e2226_ * (_e2302_ * (1.5f - (_e2250_ * (_e2302_ * _e2302_)))));
                                let _e2323_ = (_e2319_ + 0.001f);
                                let _e2324_ = (_e2320_ + 0.001f);
                                let _e2325_ = (_e2321_ + 0.001f);
                                let _e2326_ = (_e2322_ + 0.001f);
                                if (_e2323_ != _e2323_) {
                                    phi_9366_ = true;
                                } else {
                                    phi_9366_ = (0.000001f >= _e2323_);
                                }
                                let _e2330_ = phi_9366_;
                                let _e2331_ = select(_e2323_, 0.000001f, _e2330_);
                                if (_e2324_ != _e2324_) {
                                    phi_9381_ = true;
                                } else {
                                    phi_9381_ = (0.000001f >= _e2324_);
                                }
                                let _e2335_ = phi_9381_;
                                let _e2336_ = select(_e2324_, 0.000001f, _e2335_);
                                if (_e2325_ != _e2325_) {
                                    phi_9396_ = true;
                                } else {
                                    phi_9396_ = (0.000001f >= _e2325_);
                                }
                                let _e2340_ = phi_9396_;
                                let _e2341_ = select(_e2325_, 0.000001f, _e2340_);
                                if (_e2326_ != _e2326_) {
                                    phi_9411_ = true;
                                } else {
                                    phi_9411_ = (0.000001f >= _e2326_);
                                }
                                let _e2345_ = phi_9411_;
                                let _e2346_ = select(_e2326_, 0.000001f, _e2345_);
                                let _e2349_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2331_)));
                                let _e2352_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2336_)));
                                let _e2355_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2341_)));
                                let _e2358_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2346_)));
                                let _e2368_ = (_e2349_ + (_e2349_ * (1f - (_e2331_ * _e2349_))));
                                let _e2370_ = (_e2352_ + (_e2352_ * (1f - (_e2336_ * _e2352_))));
                                let _e2372_ = (_e2355_ + (_e2355_ * (1f - (_e2341_ * _e2355_))));
                                let _e2374_ = (_e2358_ + (_e2358_ * (1f - (_e2346_ * _e2358_))));
                                let _e2384_ = (_e2368_ + (_e2368_ * (1f - (_e2331_ * _e2368_))));
                                let _e2386_ = (_e2370_ + (_e2370_ * (1f - (_e2336_ * _e2370_))));
                                let _e2388_ = (_e2372_ + (_e2372_ * (1f - (_e2341_ * _e2372_))));
                                let _e2390_ = (_e2374_ + (_e2374_ * (1f - (_e2346_ * _e2374_))));
                                let _e2400_ = (_e2384_ + (_e2384_ * (1f - (_e2331_ * _e2384_))));
                                let _e2402_ = (_e2386_ + (_e2386_ * (1f - (_e2336_ * _e2386_))));
                                let _e2404_ = (_e2388_ + (_e2388_ * (1f - (_e2341_ * _e2388_))));
                                let _e2406_ = (_e2390_ + (_e2390_ * (1f - (_e2346_ * _e2390_))));
                                let _e2407_ = (_e2319_ + _e2203_);
                                let _e2408_ = (_e2320_ + _e2208_);
                                let _e2409_ = (_e2321_ + _e2213_);
                                let _e2410_ = (_e2322_ + _e2218_);
                                if (_e2407_ != _e2407_) {
                                    phi_9426_ = true;
                                } else {
                                    phi_9426_ = (0.000001f >= _e2407_);
                                }
                                let _e2414_ = phi_9426_;
                                let _e2415_ = select(_e2407_, 0.000001f, _e2414_);
                                if (_e2408_ != _e2408_) {
                                    phi_9441_ = true;
                                } else {
                                    phi_9441_ = (0.000001f >= _e2408_);
                                }
                                let _e2419_ = phi_9441_;
                                let _e2420_ = select(_e2408_, 0.000001f, _e2419_);
                                if (_e2409_ != _e2409_) {
                                    phi_9456_ = true;
                                } else {
                                    phi_9456_ = (0.000001f >= _e2409_);
                                }
                                let _e2424_ = phi_9456_;
                                let _e2425_ = select(_e2409_, 0.000001f, _e2424_);
                                if (_e2410_ != _e2410_) {
                                    phi_9471_ = true;
                                } else {
                                    phi_9471_ = (0.000001f >= _e2410_);
                                }
                                let _e2429_ = phi_9471_;
                                let _e2430_ = select(_e2410_, 0.000001f, _e2429_);
                                let _e2433_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2415_)));
                                let _e2436_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2420_)));
                                let _e2439_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2425_)));
                                let _e2442_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2430_)));
                                let _e2452_ = (_e2433_ + (_e2433_ * (1f - (_e2415_ * _e2433_))));
                                let _e2454_ = (_e2436_ + (_e2436_ * (1f - (_e2420_ * _e2436_))));
                                let _e2456_ = (_e2439_ + (_e2439_ * (1f - (_e2425_ * _e2439_))));
                                let _e2458_ = (_e2442_ + (_e2442_ * (1f - (_e2430_ * _e2442_))));
                                let _e2468_ = (_e2452_ + (_e2452_ * (1f - (_e2415_ * _e2452_))));
                                let _e2470_ = (_e2454_ + (_e2454_ * (1f - (_e2420_ * _e2454_))));
                                let _e2472_ = (_e2456_ + (_e2456_ * (1f - (_e2425_ * _e2456_))));
                                let _e2474_ = (_e2458_ + (_e2458_ * (1f - (_e2430_ * _e2458_))));
                                let _e2511_ = (_e2189_ <= 1000000000000000f);
                                let _e2513_ = (_e2192_ <= 1000000000000000f);
                                let _e2515_ = (_e2195_ <= 1000000000000000f);
                                let _e2517_ = (_e2198_ <= 1000000000000000f);
                                let _e2548_ = select(select(select(_e2189_, (0.5f * (_e2203_ + (_e2219_ * _e2400_))), _e2511_), select(-0.0005f, -(((0.0005f * (_e2203_ * (_e2468_ + (_e2468_ * (1f - (_e2415_ * _e2468_)))))) * (1f + (_e2203_ * _e2400_)))), _e2511_), (_e2124_ <= 0f)), 0f, (_e2189_ <= 0f));
                                let _e2550_ = select(select(select(_e2192_, (0.5f * (_e2208_ + (_e2220_ * _e2402_))), _e2513_), select(-0.0005f, -(((0.0005f * (_e2208_ * (_e2470_ + (_e2470_ * (1f - (_e2420_ * _e2470_)))))) * (1f + (_e2208_ * _e2402_)))), _e2513_), (_e2126_ <= 0f)), 0f, (_e2192_ <= 0f));
                                let _e2552_ = select(select(select(_e2195_, (0.5f * (_e2213_ + (_e2221_ * _e2404_))), _e2515_), select(-0.0005f, -(((0.0005f * (_e2213_ * (_e2472_ + (_e2472_ * (1f - (_e2425_ * _e2472_)))))) * (1f + (_e2213_ * _e2404_)))), _e2515_), (_e2128_ <= 0f)), 0f, (_e2195_ <= 0f));
                                let _e2554_ = select(select(select(_e2198_, (0.5f * (_e2218_ + (_e2222_ * _e2406_))), _e2517_), select(-0.0005f, -(((0.0005f * (_e2218_ * (_e2474_ + (_e2474_ * (1f - (_e2430_ * _e2474_)))))) * (1f + (_e2218_ * _e2406_)))), _e2517_), (_e2130_ <= 0f)), 0f, (_e2198_ <= 0f));
                                let _e2591_ = bitcast<u32>((((1f - (_e564_ * _e2095_)) * _e2113_) + (_e2095_ * select(((_e1503_ * _e2548_) + (_e564_ * (_e2155_ * (1f - (0.33333334f * (_e2155_ * _e2155_)))))), _e2548_, _e1949_))));
                                let _e2592_ = (_e2591_ & 2139095040u);
                                if (_e2592_ == 2139095040u) {
                                    phi_5751_ = 0u;
                                } else {
                                    if (_e2592_ == 0u) {
                                        phi_5750_ = (_e2591_ & 2147483648u);
                                    } else {
                                        phi_5750_ = _e2591_;
                                    }
                                    let _e2597_ = phi_5750_;
                                    phi_5751_ = _e2597_;
                                }
                                let _e2599_ = phi_5751_;
                                let _e2600_ = bitcast<f32>(_e2599_);
                                let _e2601_ = bitcast<u32>((((1f - (_e564_ * _e2100_)) * _e2116_) + (_e2100_ * select(((_e1503_ * _e2550_) + (_e564_ * (_e2160_ * (1f - (0.33333334f * (_e2160_ * _e2160_)))))), _e2550_, _e1949_))));
                                let _e2602_ = (_e2601_ & 2139095040u);
                                if (_e2602_ == 2139095040u) {
                                    phi_5765_ = 0u;
                                } else {
                                    if (_e2602_ == 0u) {
                                        phi_5764_ = (_e2601_ & 2147483648u);
                                    } else {
                                        phi_5764_ = _e2601_;
                                    }
                                    let _e2607_ = phi_5764_;
                                    phi_5765_ = _e2607_;
                                }
                                let _e2609_ = phi_5765_;
                                let _e2611_ = bitcast<u32>((((1f - (_e564_ * _e2105_)) * _e2119_) + (_e2105_ * select(((_e1503_ * _e2552_) + (_e564_ * (_e2165_ * (1f - (0.33333334f * (_e2165_ * _e2165_)))))), _e2552_, _e1949_))));
                                let _e2612_ = (_e2611_ & 2139095040u);
                                if (_e2612_ == 2139095040u) {
                                    phi_5779_ = 0u;
                                } else {
                                    if (_e2612_ == 0u) {
                                        phi_5778_ = (_e2611_ & 2147483648u);
                                    } else {
                                        phi_5778_ = _e2611_;
                                    }
                                    let _e2617_ = phi_5778_;
                                    phi_5779_ = _e2617_;
                                }
                                let _e2619_ = phi_5779_;
                                let _e2621_ = bitcast<u32>((((1f - (_e564_ * _e2110_)) * _e2122_) + (_e2110_ * select(((_e1503_ * _e2554_) + (_e564_ * (_e2170_ * (1f - (0.33333334f * (_e2170_ * _e2170_)))))), _e2554_, _e1949_))));
                                let _e2622_ = (_e2621_ & 2139095040u);
                                if (_e2622_ == 2139095040u) {
                                    phi_5793_ = 0u;
                                } else {
                                    if (_e2622_ == 0u) {
                                        phi_5792_ = (_e2621_ & 2147483648u);
                                    } else {
                                        phi_5792_ = _e2621_;
                                    }
                                    let _e2627_ = phi_5792_;
                                    phi_5793_ = _e2627_;
                                }
                                let _e2629_ = phi_5793_;
                                let _e2634_ = (0.5f * (_e1983_ + _e2600_));
                                let _e2641_ = (1024u + _e1370_.member_1_);
                                if (_e2641_ < 1536u) {
                                } else {
                                    phi_12137_ = true;
                                    break;
                                }
                                scan_scratch[_e2641_] = (_e1983_ + (_e570_ * (_e2634_ - _e1983_)));
                                let _e2644_ = (1280u + _e1370_.member_1_);
                                if (_e2644_ < 1536u) {
                                } else {
                                    phi_12137_ = true;
                                    break;
                                }
                                scan_scratch[_e2644_] = (_e2600_ + (_e570_ * (_e2634_ - _e2600_)));
                                phi_4114_ = type_22_(array<f32, 4>(_e2600_, bitcast<f32>(_e2609_), bitcast<f32>(_e2619_), bitcast<f32>(_e2629_)));
                                phi_4116_ = type_22_(array<f32, 4>(_e1983_, bitcast<f32>(_e1992_), bitcast<f32>(_e2002_), bitcast<f32>(_e2012_)));
                            } else {
                                phi_4114_ = type_22_();
                                phi_4116_ = type_22_();
                            }
                            let _e2648_ = phi_4114_;
                            let _e2650_ = phi_4116_;
                            phi_4110_ = _e1368_;
                            phi_4113_ = _e2648_;
                            phi_4115_ = _e2650_;
                            phi_12137_ = _e1306_;
                            if !(_e1374_) {
                                break;
                            }
                        }
                        let _e2653_ = phi_12137_;
                        phi_12422_ = _e2653_;
                        if _e2653_ {
                            break;
                        }
                        phi_12819_ = _e2653_;
                    } else {
                        phi_12819_ = _e1306_;
                    }
                    let _e2655_ = phi_12819_;
                    workgroupBarrier();
                    if _e160_ {
                        if _e885_ {
                        } else {
                            phi_12422_ = true;
                            break;
                        }
                        let _e2656_ = scan_scratch[_e884_];
                        phi_5877_ = _e2656_;
                    } else {
                        phi_5877_ = 0f;
                    }
                    let _e2658_ = phi_5877_;
                    if _e160_ {
                        if _e889_ {
                        } else {
                            phi_12422_ = true;
                            break;
                        }
                        let _e2659_ = scan_scratch[_e888_];
                        phi_5889_ = _e2659_;
                    } else {
                        phi_5889_ = 0f;
                    }
                    let _e2661_ = phi_5889_;
                    let _e2663_ = (1f - (_e775_ * _e2658_));
                    if (_e2663_ != _e2663_) {
                        phi_9606_ = true;
                    } else {
                        phi_9606_ = (0.05f >= _e2663_);
                    }
                    let _e2667_ = phi_9606_;
                    let _e2670_ = (1f - (_e775_ * _e2661_));
                    if (_e2670_ != _e2670_) {
                        phi_9621_ = true;
                    } else {
                        phi_9621_ = (0.05f >= _e2670_);
                    }
                    let _e2674_ = phi_9621_;
                    if _e160_ {
                        let _e2680_ = bitcast<u32>((((_e1108_.member * _e814_) * select(_e2663_, 0.05f, _e2667_)) + (_e853_ * _e2658_)));
                        let _e2681_ = (_e2680_ & 2139095040u);
                        if (_e2681_ == 2139095040u) {
                            phi_5913_ = true;
                        } else {
                            phi_5913_ = (_e2681_ == 0u);
                        }
                        let _e2685_ = phi_5913_;
                        phi_5919_ = bitcast<f32>(select(_e2680_, 0u, _e2685_));
                    } else {
                        phi_5919_ = 0f;
                    }
                    let _e2689_ = phi_5919_;
                    if _e160_ {
                        let _e2694_ = bitcast<u32>((((_e1345_.member * _e814_) * select(_e2670_, 0.05f, _e2674_)) + (_e853_ * _e2661_)));
                        let _e2695_ = (_e2694_ & 2139095040u);
                        if (_e2695_ == 2139095040u) {
                            phi_5937_ = true;
                        } else {
                            phi_5937_ = (_e2695_ == 0u);
                        }
                        let _e2699_ = phi_5937_;
                        phi_5943_ = bitcast<f32>(select(_e2694_, 0u, _e2699_));
                    } else {
                        phi_5943_ = 0f;
                    }
                    let _e2703_ = phi_5943_;
                    workgroupBarrier();
                    if _e869_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e867_] = _e2689_;
                    if _e873_ {
                    } else {
                        phi_12422_ = true;
                        break;
                    }
                    scan_scratch[_e872_] = _e2703_;
                    workgroupBarrier();
                    let _e2704_ = (_e114_ > 0u);
                    if _e2704_ {
                        if _e160_ {
                            let _e2705_ = (_e114_ - 1u);
                            if (_e2705_ < 1536u) {
                            } else {
                                phi_12422_ = true;
                                break;
                            }
                            let _e2708_ = scan_scratch[_e2705_];
                            phi_5972_ = _e2708_;
                        } else {
                            phi_5972_ = f32();
                        }
                        let _e2710_ = phi_5972_;
                        phi_5974_ = _e2710_;
                        phi_5975_ = select(true, false, _e160_);
                    } else {
                        phi_5974_ = f32();
                        phi_5975_ = true;
                    }
                    let _e2713_ = phi_5974_;
                    let _e2715_ = phi_5975_;
                    let _e2716_ = select(_e2713_, 0f, _e2715_);
                    if _e2704_ {
                        if _e160_ {
                            let _e2717_ = (_e114_ + 255u);
                            if (_e2717_ < 1536u) {
                            } else {
                                phi_12422_ = true;
                                break;
                            }
                            let _e2720_ = scan_scratch[_e2717_];
                            phi_5996_ = _e2720_;
                        } else {
                            phi_5996_ = f32();
                        }
                        let _e2722_ = phi_5996_;
                        phi_5998_ = _e2722_;
                        phi_5999_ = select(true, false, _e160_);
                    } else {
                        phi_5998_ = f32();
                        phi_5999_ = true;
                    }
                    let _e2725_ = phi_5998_;
                    let _e2727_ = phi_5999_;
                    let _e2728_ = select(_e2725_, 0f, _e2727_);
                    let _e2733_ = select(0f, _e2716_, (bitcast<f32>((bitcast<u32>(_e2716_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2734_ = (_e2733_ != _e2733_);
                    if _e2734_ {
                        phi_9636_ = true;
                    } else {
                        phi_9636_ = (-1000000000000000000f >= _e2733_);
                    }
                    let _e2737_ = phi_9636_;
                    let _e2738_ = select(_e2733_, -1000000000000000000f, _e2737_);
                    if _e2734_ {
                        phi_9651_ = true;
                    } else {
                        phi_9651_ = (-1000000000000000000f >= _e2733_);
                    }
                    let _e2741_ = phi_9651_;
                    let _e2742_ = select(_e2733_, -1000000000000000000f, _e2741_);
                    if _e2734_ {
                        phi_9666_ = true;
                    } else {
                        phi_9666_ = (-1000000000000000000f >= _e2733_);
                    }
                    let _e2745_ = phi_9666_;
                    let _e2746_ = select(_e2733_, -1000000000000000000f, _e2745_);
                    if _e2734_ {
                        phi_9681_ = true;
                    } else {
                        phi_9681_ = (-1000000000000000000f >= _e2733_);
                    }
                    let _e2749_ = phi_9681_;
                    let _e2750_ = select(_e2733_, -1000000000000000000f, _e2749_);
                    if (_e2738_ != _e2738_) {
                        phi_9696_ = true;
                    } else {
                        phi_9696_ = (1000000000000000000f <= _e2738_);
                    }
                    let _e2754_ = phi_9696_;
                    let _e2755_ = select(_e2738_, 1000000000000000000f, _e2754_);
                    if (_e2742_ != _e2742_) {
                        phi_9711_ = true;
                    } else {
                        phi_9711_ = (1000000000000000000f <= _e2742_);
                    }
                    let _e2759_ = phi_9711_;
                    let _e2760_ = select(_e2742_, 1000000000000000000f, _e2759_);
                    if (_e2746_ != _e2746_) {
                        phi_9726_ = true;
                    } else {
                        phi_9726_ = (1000000000000000000f <= _e2746_);
                    }
                    let _e2764_ = phi_9726_;
                    let _e2765_ = select(_e2746_, 1000000000000000000f, _e2764_);
                    if (_e2750_ != _e2750_) {
                        phi_9741_ = true;
                    } else {
                        phi_9741_ = (1000000000000000000f <= _e2750_);
                    }
                    let _e2769_ = phi_9741_;
                    let _e2770_ = select(_e2750_, 1000000000000000000f, _e2769_);
                    let _e2775_ = select(0f, _e2689_, (bitcast<f32>((bitcast<u32>(_e2689_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2776_ = (_e2775_ != _e2775_);
                    if _e2776_ {
                        phi_9756_ = true;
                    } else {
                        phi_9756_ = (-1000000000000000000f >= _e2775_);
                    }
                    let _e2779_ = phi_9756_;
                    let _e2780_ = select(_e2775_, -1000000000000000000f, _e2779_);
                    if _e2776_ {
                        phi_9771_ = true;
                    } else {
                        phi_9771_ = (-1000000000000000000f >= _e2775_);
                    }
                    let _e2783_ = phi_9771_;
                    let _e2784_ = select(_e2775_, -1000000000000000000f, _e2783_);
                    if _e2776_ {
                        phi_9786_ = true;
                    } else {
                        phi_9786_ = (-1000000000000000000f >= _e2775_);
                    }
                    let _e2787_ = phi_9786_;
                    let _e2788_ = select(_e2775_, -1000000000000000000f, _e2787_);
                    if _e2776_ {
                        phi_9801_ = true;
                    } else {
                        phi_9801_ = (-1000000000000000000f >= _e2775_);
                    }
                    let _e2791_ = phi_9801_;
                    let _e2792_ = select(_e2775_, -1000000000000000000f, _e2791_);
                    if (_e2780_ != _e2780_) {
                        phi_9816_ = true;
                    } else {
                        phi_9816_ = (1000000000000000000f <= _e2780_);
                    }
                    let _e2796_ = phi_9816_;
                    let _e2797_ = select(_e2780_, 1000000000000000000f, _e2796_);
                    if (_e2784_ != _e2784_) {
                        phi_9831_ = true;
                    } else {
                        phi_9831_ = (1000000000000000000f <= _e2784_);
                    }
                    let _e2801_ = phi_9831_;
                    let _e2802_ = select(_e2784_, 1000000000000000000f, _e2801_);
                    if (_e2788_ != _e2788_) {
                        phi_9846_ = true;
                    } else {
                        phi_9846_ = (1000000000000000000f <= _e2788_);
                    }
                    let _e2806_ = phi_9846_;
                    let _e2807_ = select(_e2788_, 1000000000000000000f, _e2806_);
                    if (_e2792_ != _e2792_) {
                        phi_9861_ = true;
                    } else {
                        phi_9861_ = (1000000000000000000f <= _e2792_);
                    }
                    let _e2811_ = phi_9861_;
                    let _e2812_ = select(_e2792_, 1000000000000000000f, _e2811_);
                    let _e2813_ = array<f32, 4>(_e2797_, _e2802_, _e2807_, _e2812_);
                    let _e2814_ = (_e2755_ != _e2755_);
                    if _e2814_ {
                        phi_9876_ = true;
                    } else {
                        phi_9876_ = (_e2797_ <= _e2755_);
                    }
                    let _e2817_ = phi_9876_;
                    let _e2818_ = select(_e2755_, _e2797_, _e2817_);
                    let _e2819_ = (_e2760_ != _e2760_);
                    if _e2819_ {
                        phi_9891_ = true;
                    } else {
                        phi_9891_ = (_e2802_ <= _e2760_);
                    }
                    let _e2822_ = phi_9891_;
                    let _e2823_ = select(_e2760_, _e2802_, _e2822_);
                    let _e2824_ = (_e2765_ != _e2765_);
                    if _e2824_ {
                        phi_9906_ = true;
                    } else {
                        phi_9906_ = (_e2807_ <= _e2765_);
                    }
                    let _e2827_ = phi_9906_;
                    let _e2828_ = select(_e2765_, _e2807_, _e2827_);
                    let _e2829_ = (_e2770_ != _e2770_);
                    if _e2829_ {
                        phi_9921_ = true;
                    } else {
                        phi_9921_ = (_e2812_ <= _e2770_);
                    }
                    let _e2832_ = phi_9921_;
                    let _e2833_ = select(_e2770_, _e2812_, _e2832_);
                    if _e2814_ {
                        phi_9936_ = true;
                    } else {
                        phi_9936_ = (_e2797_ >= _e2755_);
                    }
                    let _e2836_ = phi_9936_;
                    let _e2837_ = select(_e2755_, _e2797_, _e2836_);
                    if _e2819_ {
                        phi_9951_ = true;
                    } else {
                        phi_9951_ = (_e2802_ >= _e2760_);
                    }
                    let _e2840_ = phi_9951_;
                    let _e2841_ = select(_e2760_, _e2802_, _e2840_);
                    if _e2824_ {
                        phi_9966_ = true;
                    } else {
                        phi_9966_ = (_e2807_ >= _e2765_);
                    }
                    let _e2844_ = phi_9966_;
                    let _e2845_ = select(_e2765_, _e2807_, _e2844_);
                    if _e2829_ {
                        phi_9981_ = true;
                    } else {
                        phi_9981_ = (_e2812_ >= _e2770_);
                    }
                    let _e2848_ = phi_9981_;
                    let _e2849_ = select(_e2770_, _e2812_, _e2848_);
                    let _e2850_ = (_e2818_ != _e2818_);
                    if _e2850_ {
                        phi_9996_ = true;
                    } else {
                        phi_9996_ = (-1f >= _e2818_);
                    }
                    let _e2853_ = phi_9996_;
                    let _e2854_ = select(_e2818_, -1f, _e2853_);
                    let _e2855_ = (_e2823_ != _e2823_);
                    if _e2855_ {
                        phi_10011_ = true;
                    } else {
                        phi_10011_ = (-1f >= _e2823_);
                    }
                    let _e2858_ = phi_10011_;
                    let _e2859_ = select(_e2823_, -1f, _e2858_);
                    let _e2860_ = (_e2828_ != _e2828_);
                    if _e2860_ {
                        phi_10026_ = true;
                    } else {
                        phi_10026_ = (-1f >= _e2828_);
                    }
                    let _e2863_ = phi_10026_;
                    let _e2864_ = select(_e2828_, -1f, _e2863_);
                    let _e2865_ = (_e2833_ != _e2833_);
                    if _e2865_ {
                        phi_10041_ = true;
                    } else {
                        phi_10041_ = (-1f >= _e2833_);
                    }
                    let _e2868_ = phi_10041_;
                    let _e2869_ = select(_e2833_, -1f, _e2868_);
                    if (_e2854_ != _e2854_) {
                        phi_10056_ = true;
                    } else {
                        phi_10056_ = (1f <= _e2854_);
                    }
                    let _e2873_ = phi_10056_;
                    let _e2874_ = select(_e2854_, 1f, _e2873_);
                    if (_e2859_ != _e2859_) {
                        phi_10071_ = true;
                    } else {
                        phi_10071_ = (1f <= _e2859_);
                    }
                    let _e2878_ = phi_10071_;
                    let _e2879_ = select(_e2859_, 1f, _e2878_);
                    if (_e2864_ != _e2864_) {
                        phi_10086_ = true;
                    } else {
                        phi_10086_ = (1f <= _e2864_);
                    }
                    let _e2883_ = phi_10086_;
                    let _e2884_ = select(_e2864_, 1f, _e2883_);
                    if (_e2869_ != _e2869_) {
                        phi_10101_ = true;
                    } else {
                        phi_10101_ = (1f <= _e2869_);
                    }
                    let _e2888_ = phi_10101_;
                    let _e2889_ = select(_e2869_, 1f, _e2888_);
                    let _e2890_ = (_e2837_ != _e2837_);
                    if _e2890_ {
                        phi_10116_ = true;
                    } else {
                        phi_10116_ = (-1f >= _e2837_);
                    }
                    let _e2893_ = phi_10116_;
                    let _e2894_ = select(_e2837_, -1f, _e2893_);
                    let _e2895_ = (_e2841_ != _e2841_);
                    if _e2895_ {
                        phi_10131_ = true;
                    } else {
                        phi_10131_ = (-1f >= _e2841_);
                    }
                    let _e2898_ = phi_10131_;
                    let _e2899_ = select(_e2841_, -1f, _e2898_);
                    let _e2900_ = (_e2845_ != _e2845_);
                    if _e2900_ {
                        phi_10146_ = true;
                    } else {
                        phi_10146_ = (-1f >= _e2845_);
                    }
                    let _e2903_ = phi_10146_;
                    let _e2904_ = select(_e2845_, -1f, _e2903_);
                    let _e2905_ = (_e2849_ != _e2849_);
                    if _e2905_ {
                        phi_10161_ = true;
                    } else {
                        phi_10161_ = (-1f >= _e2849_);
                    }
                    let _e2908_ = phi_10161_;
                    let _e2909_ = select(_e2849_, -1f, _e2908_);
                    if (_e2894_ != _e2894_) {
                        phi_10176_ = true;
                    } else {
                        phi_10176_ = (1f <= _e2894_);
                    }
                    let _e2913_ = phi_10176_;
                    let _e2914_ = select(_e2894_, 1f, _e2913_);
                    if (_e2899_ != _e2899_) {
                        phi_10191_ = true;
                    } else {
                        phi_10191_ = (1f <= _e2899_);
                    }
                    let _e2918_ = phi_10191_;
                    let _e2919_ = select(_e2899_, 1f, _e2918_);
                    if (_e2904_ != _e2904_) {
                        phi_10206_ = true;
                    } else {
                        phi_10206_ = (1f <= _e2904_);
                    }
                    let _e2923_ = phi_10206_;
                    let _e2924_ = select(_e2904_, 1f, _e2923_);
                    if (_e2909_ != _e2909_) {
                        phi_10221_ = true;
                    } else {
                        phi_10221_ = (1f <= _e2909_);
                    }
                    let _e2928_ = phi_10221_;
                    let _e2929_ = select(_e2909_, 1f, _e2928_);
                    let _e2934_ = (0.5f * (_e2914_ + _e2874_));
                    let _e2935_ = (0.5f * (_e2919_ + _e2879_));
                    let _e2936_ = (0.5f * (_e2924_ + _e2884_));
                    let _e2937_ = (0.5f * (_e2929_ + _e2889_));
                    let _e2938_ = (_e2914_ - _e2874_);
                    let _e2939_ = (_e2919_ - _e2879_);
                    let _e2940_ = (_e2924_ - _e2884_);
                    let _e2941_ = (_e2929_ - _e2889_);
                    let _e2942_ = (0.5f * _e2938_);
                    let _e2943_ = (0.5f * _e2939_);
                    let _e2944_ = (0.5f * _e2940_);
                    let _e2945_ = (0.5f * _e2941_);
                    let _e2946_ = (_e2934_ * _e2934_);
                    let _e2947_ = (_e2935_ * _e2935_);
                    let _e2948_ = (_e2936_ * _e2936_);
                    let _e2949_ = (_e2937_ * _e2937_);
                    let _e2950_ = (_e2942_ * _e2942_);
                    let _e2951_ = (_e2943_ * _e2943_);
                    let _e2952_ = (_e2944_ * _e2944_);
                    let _e2953_ = (_e2945_ * _e2945_);
                    if (_e2938_ != _e2938_) {
                        phi_10236_ = true;
                    } else {
                        phi_10236_ = (0f >= _e2938_);
                    }
                    let _e2957_ = phi_10236_;
                    let _e2958_ = select(_e2938_, 0f, _e2957_);
                    if (_e2939_ != _e2939_) {
                        phi_10251_ = true;
                    } else {
                        phi_10251_ = (0f >= _e2939_);
                    }
                    let _e2962_ = phi_10251_;
                    let _e2963_ = select(_e2939_, 0f, _e2962_);
                    if (_e2940_ != _e2940_) {
                        phi_10266_ = true;
                    } else {
                        phi_10266_ = (0f >= _e2940_);
                    }
                    let _e2967_ = phi_10266_;
                    let _e2968_ = select(_e2940_, 0f, _e2967_);
                    if (_e2941_ != _e2941_) {
                        phi_10281_ = true;
                    } else {
                        phi_10281_ = (0f >= _e2941_);
                    }
                    let _e2972_ = phi_10281_;
                    let _e2973_ = select(_e2941_, 0f, _e2972_);
                    let _e2974_ = (_e2837_ - _e2818_);
                    let _e2975_ = (_e2841_ - _e2823_);
                    let _e2976_ = (_e2845_ - _e2828_);
                    let _e2977_ = (_e2849_ - _e2833_);
                    if _e2890_ {
                        phi_10296_ = true;
                    } else {
                        phi_10296_ = (-1f <= _e2837_);
                    }
                    let _e2980_ = phi_10296_;
                    if _e2895_ {
                        phi_10311_ = true;
                    } else {
                        phi_10311_ = (-1f <= _e2841_);
                    }
                    let _e2984_ = phi_10311_;
                    if _e2900_ {
                        phi_10326_ = true;
                    } else {
                        phi_10326_ = (-1f <= _e2845_);
                    }
                    let _e2988_ = phi_10326_;
                    if _e2905_ {
                        phi_10341_ = true;
                    } else {
                        phi_10341_ = (-1f <= _e2849_);
                    }
                    let _e2992_ = phi_10341_;
                    let _e2994_ = (select(_e2837_, -1f, _e2980_) - _e2818_);
                    let _e2995_ = (select(_e2841_, -1f, _e2984_) - _e2823_);
                    let _e2996_ = (select(_e2845_, -1f, _e2988_) - _e2828_);
                    let _e2997_ = (select(_e2849_, -1f, _e2992_) - _e2833_);
                    if (_e2994_ != _e2994_) {
                        phi_10356_ = true;
                    } else {
                        phi_10356_ = (0f >= _e2994_);
                    }
                    let _e3001_ = phi_10356_;
                    if (_e2995_ != _e2995_) {
                        phi_10371_ = true;
                    } else {
                        phi_10371_ = (0f >= _e2995_);
                    }
                    let _e3006_ = phi_10371_;
                    if (_e2996_ != _e2996_) {
                        phi_10386_ = true;
                    } else {
                        phi_10386_ = (0f >= _e2996_);
                    }
                    let _e3011_ = phi_10386_;
                    if (_e2997_ != _e2997_) {
                        phi_10401_ = true;
                    } else {
                        phi_10401_ = (0f >= _e2997_);
                    }
                    let _e3016_ = phi_10401_;
                    if _e2850_ {
                        phi_10416_ = true;
                    } else {
                        phi_10416_ = (1f >= _e2818_);
                    }
                    let _e3020_ = phi_10416_;
                    if _e2855_ {
                        phi_10431_ = true;
                    } else {
                        phi_10431_ = (1f >= _e2823_);
                    }
                    let _e3024_ = phi_10431_;
                    if _e2860_ {
                        phi_10446_ = true;
                    } else {
                        phi_10446_ = (1f >= _e2828_);
                    }
                    let _e3028_ = phi_10446_;
                    if _e2865_ {
                        phi_10461_ = true;
                    } else {
                        phi_10461_ = (1f >= _e2833_);
                    }
                    let _e3032_ = phi_10461_;
                    let _e3034_ = (_e2837_ - select(_e2818_, 1f, _e3020_));
                    let _e3035_ = (_e2841_ - select(_e2823_, 1f, _e3024_));
                    let _e3036_ = (_e2845_ - select(_e2828_, 1f, _e3028_));
                    let _e3037_ = (_e2849_ - select(_e2833_, 1f, _e3032_));
                    if (_e3034_ != _e3034_) {
                        phi_10476_ = true;
                    } else {
                        phi_10476_ = (0f >= _e3034_);
                    }
                    let _e3041_ = phi_10476_;
                    if (_e3035_ != _e3035_) {
                        phi_10491_ = true;
                    } else {
                        phi_10491_ = (0f >= _e3035_);
                    }
                    let _e3046_ = phi_10491_;
                    if (_e3036_ != _e3036_) {
                        phi_10506_ = true;
                    } else {
                        phi_10506_ = (0f >= _e3036_);
                    }
                    let _e3051_ = phi_10506_;
                    if (_e3037_ != _e3037_) {
                        phi_10521_ = true;
                    } else {
                        phi_10521_ = (0f >= _e3037_);
                    }
                    let _e3056_ = phi_10521_;
                    let _e3060_ = bitcast<f32>((bitcast<u32>(_e2755_) & 2147483647u));
                    let _e3063_ = bitcast<f32>((bitcast<u32>(_e2760_) & 2147483647u));
                    let _e3066_ = bitcast<f32>((bitcast<u32>(_e2765_) & 2147483647u));
                    let _e3069_ = bitcast<f32>((bitcast<u32>(_e2770_) & 2147483647u));
                    let _e3072_ = bitcast<f32>((bitcast<u32>(_e2797_) & 2147483647u));
                    let _e3075_ = bitcast<f32>((bitcast<u32>(_e2802_) & 2147483647u));
                    let _e3078_ = bitcast<f32>((bitcast<u32>(_e2807_) & 2147483647u));
                    let _e3081_ = bitcast<f32>((bitcast<u32>(_e2812_) & 2147483647u));
                    if (_e3060_ != _e3060_) {
                        phi_10536_ = true;
                    } else {
                        phi_10536_ = (_e3072_ >= _e3060_);
                    }
                    let _e3085_ = phi_10536_;
                    let _e3086_ = select(_e3060_, _e3072_, _e3085_);
                    if (_e3063_ != _e3063_) {
                        phi_10551_ = true;
                    } else {
                        phi_10551_ = (_e3075_ >= _e3063_);
                    }
                    let _e3090_ = phi_10551_;
                    let _e3091_ = select(_e3063_, _e3075_, _e3090_);
                    if (_e3066_ != _e3066_) {
                        phi_10566_ = true;
                    } else {
                        phi_10566_ = (_e3078_ >= _e3066_);
                    }
                    let _e3095_ = phi_10566_;
                    let _e3096_ = select(_e3066_, _e3078_, _e3095_);
                    if (_e3069_ != _e3069_) {
                        phi_10581_ = true;
                    } else {
                        phi_10581_ = (_e3081_ >= _e3069_);
                    }
                    let _e3100_ = phi_10581_;
                    let _e3101_ = select(_e3069_, _e3081_, _e3100_);
                    let _e3118_ = (_e3086_ <= 0f);
                    let _e3119_ = select((_e2934_ * (1.5f - (0.5f * (_e2946_ + _e2950_)))), 0f, _e3118_);
                    let _e3120_ = (_e3091_ <= 0f);
                    let _e3121_ = select((_e2935_ * (1.5f - (0.5f * (_e2947_ + _e2951_)))), 0f, _e3120_);
                    let _e3122_ = (_e3096_ <= 0f);
                    let _e3123_ = select((_e2936_ * (1.5f - (0.5f * (_e2948_ + _e2952_)))), 0f, _e3122_);
                    let _e3124_ = (_e3101_ <= 0f);
                    let _e3125_ = select((_e2937_ * (1.5f - (0.5f * (_e2949_ + _e2953_)))), 0f, _e3124_);
                    let _e3167_ = select(((_e2946_ + (0.33333334f * _e2950_)) - (((_e2946_ * _e2946_) + ((2f * _e2946_) * _e2950_)) + (0.2f * (_e2950_ * _e2950_)))), 0f, _e3118_);
                    let _e3168_ = select(((_e2947_ + (0.33333334f * _e2951_)) - (((_e2947_ * _e2947_) + ((2f * _e2947_) * _e2951_)) + (0.2f * (_e2951_ * _e2951_)))), 0f, _e3120_);
                    let _e3169_ = select(((_e2948_ + (0.33333334f * _e2952_)) - (((_e2948_ * _e2948_) + ((2f * _e2948_) * _e2952_)) + (0.2f * (_e2952_ * _e2952_)))), 0f, _e3122_);
                    let _e3170_ = select(((_e2949_ + (0.33333334f * _e2953_)) - (((_e2949_ * _e2949_) + ((2f * _e2949_) * _e2953_)) + (0.2f * (_e2953_ * _e2953_)))), 0f, _e3124_);
                    if (_e3086_ <= 1f) {
                        if (_e3091_ <= 1f) {
                            let _e3174_ = (_e3096_ <= 1f);
                            if _e3174_ {
                                phi_6426_ = (_e3101_ <= 1f);
                            } else {
                                phi_6426_ = bool();
                            }
                            let _e3177_ = phi_6426_;
                            phi_6428_ = _e3177_;
                            phi_6429_ = select(true, false, _e3174_);
                        } else {
                            phi_6428_ = bool();
                            phi_6429_ = true;
                        }
                        let _e3180_ = phi_6428_;
                        let _e3182_ = phi_6429_;
                        phi_6430_ = _e3180_;
                        phi_6431_ = _e3182_;
                    } else {
                        phi_6430_ = bool();
                        phi_6431_ = true;
                    }
                    let _e3184_ = phi_6430_;
                    let _e3186_ = phi_6431_;
                    if select(_e3184_, false, _e3186_) {
                        phi_6809_ = type_27_(type_22_(array<f32, 4>(_e3119_, _e3121_, _e3123_, _e3125_)), type_22_(array<f32, 4>(_e3167_, _e3168_, _e3169_, _e3170_)), type_22_(_e2813_));
                    } else {
                        if (_e2974_ != _e2974_) {
                            phi_10596_ = true;
                        } else {
                            phi_10596_ = (0.000001f >= _e2974_);
                        }
                        let _e3191_ = phi_10596_;
                        let _e3192_ = select(_e2974_, 0.000001f, _e3191_);
                        if (_e2975_ != _e2975_) {
                            phi_10611_ = true;
                        } else {
                            phi_10611_ = (0.000001f >= _e2975_);
                        }
                        let _e3196_ = phi_10611_;
                        let _e3197_ = select(_e2975_, 0.000001f, _e3196_);
                        if (_e2976_ != _e2976_) {
                            phi_10626_ = true;
                        } else {
                            phi_10626_ = (0.000001f >= _e2976_);
                        }
                        let _e3201_ = phi_10626_;
                        let _e3202_ = select(_e2976_, 0.000001f, _e3201_);
                        if (_e2977_ != _e2977_) {
                            phi_10641_ = true;
                        } else {
                            phi_10641_ = (0.000001f >= _e2977_);
                        }
                        let _e3206_ = phi_10641_;
                        let _e3207_ = select(_e2977_, 0.000001f, _e3206_);
                        let _e3210_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3192_)));
                        let _e3213_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3197_)));
                        let _e3216_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3202_)));
                        let _e3219_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3207_)));
                        let _e3229_ = (_e3210_ + (_e3210_ * (1f - (_e3192_ * _e3210_))));
                        let _e3231_ = (_e3213_ + (_e3213_ * (1f - (_e3197_ * _e3213_))));
                        let _e3233_ = (_e3216_ + (_e3216_ * (1f - (_e3202_ * _e3216_))));
                        let _e3235_ = (_e3219_ + (_e3219_ * (1f - (_e3207_ * _e3219_))));
                        let _e3245_ = (_e3229_ + (_e3229_ * (1f - (_e3192_ * _e3229_))));
                        let _e3247_ = (_e3231_ + (_e3231_ * (1f - (_e3197_ * _e3231_))));
                        let _e3249_ = (_e3233_ + (_e3233_ * (1f - (_e3202_ * _e3233_))));
                        let _e3251_ = (_e3235_ + (_e3235_ * (1f - (_e3207_ * _e3235_))));
                        let _e3304_ = (select((select(0f, 1f, (-1f <= _e2818_)) * select(0f, 1f, (_e2837_ <= 1f))), 1f, (_e2974_ <= 0.000001f)) <= 0f);
                        let _e3306_ = (select((select(0f, 1f, (-1f <= _e2823_)) * select(0f, 1f, (_e2841_ <= 1f))), 1f, (_e2975_ <= 0.000001f)) <= 0f);
                        let _e3308_ = (select((select(0f, 1f, (-1f <= _e2828_)) * select(0f, 1f, (_e2845_ <= 1f))), 1f, (_e2976_ <= 0.000001f)) <= 0f);
                        let _e3310_ = (select((select(0f, 1f, (-1f <= _e2833_)) * select(0f, 1f, (_e2849_ <= 1f))), 1f, (_e2977_ <= 0.000001f)) <= 0f);
                        let _e3316_ = (1f <= _e2818_);
                        let _e3318_ = (1f <= _e2823_);
                        let _e3320_ = (1f <= _e2828_);
                        let _e3322_ = (1f <= _e2833_);
                        let _e3324_ = (_e2837_ <= -1f);
                        let _e3326_ = (_e2841_ <= -1f);
                        let _e3328_ = (_e2845_ <= -1f);
                        let _e3330_ = (_e2849_ <= -1f);
                        phi_6809_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3119_, (((select(_e3034_, 0f, _e3041_) - select(_e2994_, 0f, _e3001_)) + (_e2958_ * _e3119_)) * _e3245_), _e3304_), 1f, _e3316_), -1f, _e3324_), select(select(select(_e3121_, (((select(_e3035_, 0f, _e3046_) - select(_e2995_, 0f, _e3006_)) + (_e2963_ * _e3121_)) * _e3247_), _e3306_), 1f, _e3318_), -1f, _e3326_), select(select(select(_e3123_, (((select(_e3036_, 0f, _e3051_) - select(_e2996_, 0f, _e3011_)) + (_e2968_ * _e3123_)) * _e3249_), _e3308_), 1f, _e3320_), -1f, _e3328_), select(select(select(_e3125_, (((select(_e3037_, 0f, _e3056_) - select(_e2997_, 0f, _e3016_)) + (_e2973_ * _e3125_)) * _e3251_), _e3310_), 1f, _e3322_), -1f, _e3330_))), type_22_(array<f32, 4>(select(select(select(_e3167_, ((_e2958_ * _e3167_) * _e3245_), _e3304_), 0f, _e3316_), 0f, _e3324_), select(select(select(_e3168_, ((_e2963_ * _e3168_) * _e3247_), _e3306_), 0f, _e3318_), 0f, _e3326_), select(select(select(_e3169_, ((_e2968_ * _e3169_) * _e3249_), _e3308_), 0f, _e3320_), 0f, _e3328_), select(select(select(_e3170_, ((_e2973_ * _e3170_) * _e3251_), _e3310_), 0f, _e3322_), 0f, _e3330_))), type_22_(_e2813_));
                    }
                    let _e3351_ = phi_6809_;
                    let _e3358_ = select(0f, _e2728_, (bitcast<f32>((bitcast<u32>(_e2728_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3359_ = (_e3358_ != _e3358_);
                    if _e3359_ {
                        phi_10656_ = true;
                    } else {
                        phi_10656_ = (-1000000000000000000f >= _e3358_);
                    }
                    let _e3362_ = phi_10656_;
                    let _e3363_ = select(_e3358_, -1000000000000000000f, _e3362_);
                    if _e3359_ {
                        phi_10671_ = true;
                    } else {
                        phi_10671_ = (-1000000000000000000f >= _e3358_);
                    }
                    let _e3366_ = phi_10671_;
                    let _e3367_ = select(_e3358_, -1000000000000000000f, _e3366_);
                    if _e3359_ {
                        phi_10686_ = true;
                    } else {
                        phi_10686_ = (-1000000000000000000f >= _e3358_);
                    }
                    let _e3370_ = phi_10686_;
                    let _e3371_ = select(_e3358_, -1000000000000000000f, _e3370_);
                    if _e3359_ {
                        phi_10701_ = true;
                    } else {
                        phi_10701_ = (-1000000000000000000f >= _e3358_);
                    }
                    let _e3374_ = phi_10701_;
                    let _e3375_ = select(_e3358_, -1000000000000000000f, _e3374_);
                    if (_e3363_ != _e3363_) {
                        phi_10716_ = true;
                    } else {
                        phi_10716_ = (1000000000000000000f <= _e3363_);
                    }
                    let _e3379_ = phi_10716_;
                    let _e3380_ = select(_e3363_, 1000000000000000000f, _e3379_);
                    if (_e3367_ != _e3367_) {
                        phi_10731_ = true;
                    } else {
                        phi_10731_ = (1000000000000000000f <= _e3367_);
                    }
                    let _e3384_ = phi_10731_;
                    let _e3385_ = select(_e3367_, 1000000000000000000f, _e3384_);
                    if (_e3371_ != _e3371_) {
                        phi_10746_ = true;
                    } else {
                        phi_10746_ = (1000000000000000000f <= _e3371_);
                    }
                    let _e3389_ = phi_10746_;
                    let _e3390_ = select(_e3371_, 1000000000000000000f, _e3389_);
                    if (_e3375_ != _e3375_) {
                        phi_10761_ = true;
                    } else {
                        phi_10761_ = (1000000000000000000f <= _e3375_);
                    }
                    let _e3394_ = phi_10761_;
                    let _e3395_ = select(_e3375_, 1000000000000000000f, _e3394_);
                    let _e3400_ = select(0f, _e2703_, (bitcast<f32>((bitcast<u32>(_e2703_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3401_ = (_e3400_ != _e3400_);
                    if _e3401_ {
                        phi_10776_ = true;
                    } else {
                        phi_10776_ = (-1000000000000000000f >= _e3400_);
                    }
                    let _e3404_ = phi_10776_;
                    let _e3405_ = select(_e3400_, -1000000000000000000f, _e3404_);
                    if _e3401_ {
                        phi_10791_ = true;
                    } else {
                        phi_10791_ = (-1000000000000000000f >= _e3400_);
                    }
                    let _e3408_ = phi_10791_;
                    let _e3409_ = select(_e3400_, -1000000000000000000f, _e3408_);
                    if _e3401_ {
                        phi_10806_ = true;
                    } else {
                        phi_10806_ = (-1000000000000000000f >= _e3400_);
                    }
                    let _e3412_ = phi_10806_;
                    let _e3413_ = select(_e3400_, -1000000000000000000f, _e3412_);
                    if _e3401_ {
                        phi_10821_ = true;
                    } else {
                        phi_10821_ = (-1000000000000000000f >= _e3400_);
                    }
                    let _e3416_ = phi_10821_;
                    let _e3417_ = select(_e3400_, -1000000000000000000f, _e3416_);
                    if (_e3405_ != _e3405_) {
                        phi_10836_ = true;
                    } else {
                        phi_10836_ = (1000000000000000000f <= _e3405_);
                    }
                    let _e3421_ = phi_10836_;
                    let _e3422_ = select(_e3405_, 1000000000000000000f, _e3421_);
                    if (_e3409_ != _e3409_) {
                        phi_10851_ = true;
                    } else {
                        phi_10851_ = (1000000000000000000f <= _e3409_);
                    }
                    let _e3426_ = phi_10851_;
                    let _e3427_ = select(_e3409_, 1000000000000000000f, _e3426_);
                    if (_e3413_ != _e3413_) {
                        phi_10866_ = true;
                    } else {
                        phi_10866_ = (1000000000000000000f <= _e3413_);
                    }
                    let _e3431_ = phi_10866_;
                    let _e3432_ = select(_e3413_, 1000000000000000000f, _e3431_);
                    if (_e3417_ != _e3417_) {
                        phi_10881_ = true;
                    } else {
                        phi_10881_ = (1000000000000000000f <= _e3417_);
                    }
                    let _e3436_ = phi_10881_;
                    let _e3437_ = select(_e3417_, 1000000000000000000f, _e3436_);
                    let _e3438_ = array<f32, 4>(_e3422_, _e3427_, _e3432_, _e3437_);
                    let _e3439_ = (_e3380_ != _e3380_);
                    if _e3439_ {
                        phi_10896_ = true;
                    } else {
                        phi_10896_ = (_e3422_ <= _e3380_);
                    }
                    let _e3442_ = phi_10896_;
                    let _e3443_ = select(_e3380_, _e3422_, _e3442_);
                    let _e3444_ = (_e3385_ != _e3385_);
                    if _e3444_ {
                        phi_10911_ = true;
                    } else {
                        phi_10911_ = (_e3427_ <= _e3385_);
                    }
                    let _e3447_ = phi_10911_;
                    let _e3448_ = select(_e3385_, _e3427_, _e3447_);
                    let _e3449_ = (_e3390_ != _e3390_);
                    if _e3449_ {
                        phi_10926_ = true;
                    } else {
                        phi_10926_ = (_e3432_ <= _e3390_);
                    }
                    let _e3452_ = phi_10926_;
                    let _e3453_ = select(_e3390_, _e3432_, _e3452_);
                    let _e3454_ = (_e3395_ != _e3395_);
                    if _e3454_ {
                        phi_10941_ = true;
                    } else {
                        phi_10941_ = (_e3437_ <= _e3395_);
                    }
                    let _e3457_ = phi_10941_;
                    let _e3458_ = select(_e3395_, _e3437_, _e3457_);
                    if _e3439_ {
                        phi_10956_ = true;
                    } else {
                        phi_10956_ = (_e3422_ >= _e3380_);
                    }
                    let _e3461_ = phi_10956_;
                    let _e3462_ = select(_e3380_, _e3422_, _e3461_);
                    if _e3444_ {
                        phi_10971_ = true;
                    } else {
                        phi_10971_ = (_e3427_ >= _e3385_);
                    }
                    let _e3465_ = phi_10971_;
                    let _e3466_ = select(_e3385_, _e3427_, _e3465_);
                    if _e3449_ {
                        phi_10986_ = true;
                    } else {
                        phi_10986_ = (_e3432_ >= _e3390_);
                    }
                    let _e3469_ = phi_10986_;
                    let _e3470_ = select(_e3390_, _e3432_, _e3469_);
                    if _e3454_ {
                        phi_11001_ = true;
                    } else {
                        phi_11001_ = (_e3437_ >= _e3395_);
                    }
                    let _e3473_ = phi_11001_;
                    let _e3474_ = select(_e3395_, _e3437_, _e3473_);
                    let _e3475_ = (_e3443_ != _e3443_);
                    if _e3475_ {
                        phi_11016_ = true;
                    } else {
                        phi_11016_ = (-1f >= _e3443_);
                    }
                    let _e3478_ = phi_11016_;
                    let _e3479_ = select(_e3443_, -1f, _e3478_);
                    let _e3480_ = (_e3448_ != _e3448_);
                    if _e3480_ {
                        phi_11031_ = true;
                    } else {
                        phi_11031_ = (-1f >= _e3448_);
                    }
                    let _e3483_ = phi_11031_;
                    let _e3484_ = select(_e3448_, -1f, _e3483_);
                    let _e3485_ = (_e3453_ != _e3453_);
                    if _e3485_ {
                        phi_11046_ = true;
                    } else {
                        phi_11046_ = (-1f >= _e3453_);
                    }
                    let _e3488_ = phi_11046_;
                    let _e3489_ = select(_e3453_, -1f, _e3488_);
                    let _e3490_ = (_e3458_ != _e3458_);
                    if _e3490_ {
                        phi_11061_ = true;
                    } else {
                        phi_11061_ = (-1f >= _e3458_);
                    }
                    let _e3493_ = phi_11061_;
                    let _e3494_ = select(_e3458_, -1f, _e3493_);
                    if (_e3479_ != _e3479_) {
                        phi_11076_ = true;
                    } else {
                        phi_11076_ = (1f <= _e3479_);
                    }
                    let _e3498_ = phi_11076_;
                    let _e3499_ = select(_e3479_, 1f, _e3498_);
                    if (_e3484_ != _e3484_) {
                        phi_11091_ = true;
                    } else {
                        phi_11091_ = (1f <= _e3484_);
                    }
                    let _e3503_ = phi_11091_;
                    let _e3504_ = select(_e3484_, 1f, _e3503_);
                    if (_e3489_ != _e3489_) {
                        phi_11106_ = true;
                    } else {
                        phi_11106_ = (1f <= _e3489_);
                    }
                    let _e3508_ = phi_11106_;
                    let _e3509_ = select(_e3489_, 1f, _e3508_);
                    if (_e3494_ != _e3494_) {
                        phi_11121_ = true;
                    } else {
                        phi_11121_ = (1f <= _e3494_);
                    }
                    let _e3513_ = phi_11121_;
                    let _e3514_ = select(_e3494_, 1f, _e3513_);
                    let _e3515_ = (_e3462_ != _e3462_);
                    if _e3515_ {
                        phi_11136_ = true;
                    } else {
                        phi_11136_ = (-1f >= _e3462_);
                    }
                    let _e3518_ = phi_11136_;
                    let _e3519_ = select(_e3462_, -1f, _e3518_);
                    let _e3520_ = (_e3466_ != _e3466_);
                    if _e3520_ {
                        phi_11151_ = true;
                    } else {
                        phi_11151_ = (-1f >= _e3466_);
                    }
                    let _e3523_ = phi_11151_;
                    let _e3524_ = select(_e3466_, -1f, _e3523_);
                    let _e3525_ = (_e3470_ != _e3470_);
                    if _e3525_ {
                        phi_11166_ = true;
                    } else {
                        phi_11166_ = (-1f >= _e3470_);
                    }
                    let _e3528_ = phi_11166_;
                    let _e3529_ = select(_e3470_, -1f, _e3528_);
                    let _e3530_ = (_e3474_ != _e3474_);
                    if _e3530_ {
                        phi_11181_ = true;
                    } else {
                        phi_11181_ = (-1f >= _e3474_);
                    }
                    let _e3533_ = phi_11181_;
                    let _e3534_ = select(_e3474_, -1f, _e3533_);
                    if (_e3519_ != _e3519_) {
                        phi_11196_ = true;
                    } else {
                        phi_11196_ = (1f <= _e3519_);
                    }
                    let _e3538_ = phi_11196_;
                    let _e3539_ = select(_e3519_, 1f, _e3538_);
                    if (_e3524_ != _e3524_) {
                        phi_11211_ = true;
                    } else {
                        phi_11211_ = (1f <= _e3524_);
                    }
                    let _e3543_ = phi_11211_;
                    let _e3544_ = select(_e3524_, 1f, _e3543_);
                    if (_e3529_ != _e3529_) {
                        phi_11226_ = true;
                    } else {
                        phi_11226_ = (1f <= _e3529_);
                    }
                    let _e3548_ = phi_11226_;
                    let _e3549_ = select(_e3529_, 1f, _e3548_);
                    if (_e3534_ != _e3534_) {
                        phi_11241_ = true;
                    } else {
                        phi_11241_ = (1f <= _e3534_);
                    }
                    let _e3553_ = phi_11241_;
                    let _e3554_ = select(_e3534_, 1f, _e3553_);
                    let _e3559_ = (0.5f * (_e3539_ + _e3499_));
                    let _e3560_ = (0.5f * (_e3544_ + _e3504_));
                    let _e3561_ = (0.5f * (_e3549_ + _e3509_));
                    let _e3562_ = (0.5f * (_e3554_ + _e3514_));
                    let _e3563_ = (_e3539_ - _e3499_);
                    let _e3564_ = (_e3544_ - _e3504_);
                    let _e3565_ = (_e3549_ - _e3509_);
                    let _e3566_ = (_e3554_ - _e3514_);
                    let _e3567_ = (0.5f * _e3563_);
                    let _e3568_ = (0.5f * _e3564_);
                    let _e3569_ = (0.5f * _e3565_);
                    let _e3570_ = (0.5f * _e3566_);
                    let _e3571_ = (_e3559_ * _e3559_);
                    let _e3572_ = (_e3560_ * _e3560_);
                    let _e3573_ = (_e3561_ * _e3561_);
                    let _e3574_ = (_e3562_ * _e3562_);
                    let _e3575_ = (_e3567_ * _e3567_);
                    let _e3576_ = (_e3568_ * _e3568_);
                    let _e3577_ = (_e3569_ * _e3569_);
                    let _e3578_ = (_e3570_ * _e3570_);
                    if (_e3563_ != _e3563_) {
                        phi_11256_ = true;
                    } else {
                        phi_11256_ = (0f >= _e3563_);
                    }
                    let _e3582_ = phi_11256_;
                    let _e3583_ = select(_e3563_, 0f, _e3582_);
                    if (_e3564_ != _e3564_) {
                        phi_11271_ = true;
                    } else {
                        phi_11271_ = (0f >= _e3564_);
                    }
                    let _e3587_ = phi_11271_;
                    let _e3588_ = select(_e3564_, 0f, _e3587_);
                    if (_e3565_ != _e3565_) {
                        phi_11286_ = true;
                    } else {
                        phi_11286_ = (0f >= _e3565_);
                    }
                    let _e3592_ = phi_11286_;
                    let _e3593_ = select(_e3565_, 0f, _e3592_);
                    if (_e3566_ != _e3566_) {
                        phi_11301_ = true;
                    } else {
                        phi_11301_ = (0f >= _e3566_);
                    }
                    let _e3597_ = phi_11301_;
                    let _e3598_ = select(_e3566_, 0f, _e3597_);
                    let _e3599_ = (_e3462_ - _e3443_);
                    let _e3600_ = (_e3466_ - _e3448_);
                    let _e3601_ = (_e3470_ - _e3453_);
                    let _e3602_ = (_e3474_ - _e3458_);
                    if _e3515_ {
                        phi_11316_ = true;
                    } else {
                        phi_11316_ = (-1f <= _e3462_);
                    }
                    let _e3605_ = phi_11316_;
                    if _e3520_ {
                        phi_11331_ = true;
                    } else {
                        phi_11331_ = (-1f <= _e3466_);
                    }
                    let _e3609_ = phi_11331_;
                    if _e3525_ {
                        phi_11346_ = true;
                    } else {
                        phi_11346_ = (-1f <= _e3470_);
                    }
                    let _e3613_ = phi_11346_;
                    if _e3530_ {
                        phi_11361_ = true;
                    } else {
                        phi_11361_ = (-1f <= _e3474_);
                    }
                    let _e3617_ = phi_11361_;
                    let _e3619_ = (select(_e3462_, -1f, _e3605_) - _e3443_);
                    let _e3620_ = (select(_e3466_, -1f, _e3609_) - _e3448_);
                    let _e3621_ = (select(_e3470_, -1f, _e3613_) - _e3453_);
                    let _e3622_ = (select(_e3474_, -1f, _e3617_) - _e3458_);
                    if (_e3619_ != _e3619_) {
                        phi_11376_ = true;
                    } else {
                        phi_11376_ = (0f >= _e3619_);
                    }
                    let _e3626_ = phi_11376_;
                    if (_e3620_ != _e3620_) {
                        phi_11391_ = true;
                    } else {
                        phi_11391_ = (0f >= _e3620_);
                    }
                    let _e3631_ = phi_11391_;
                    if (_e3621_ != _e3621_) {
                        phi_11406_ = true;
                    } else {
                        phi_11406_ = (0f >= _e3621_);
                    }
                    let _e3636_ = phi_11406_;
                    if (_e3622_ != _e3622_) {
                        phi_11421_ = true;
                    } else {
                        phi_11421_ = (0f >= _e3622_);
                    }
                    let _e3641_ = phi_11421_;
                    if _e3475_ {
                        phi_11436_ = true;
                    } else {
                        phi_11436_ = (1f >= _e3443_);
                    }
                    let _e3645_ = phi_11436_;
                    if _e3480_ {
                        phi_11451_ = true;
                    } else {
                        phi_11451_ = (1f >= _e3448_);
                    }
                    let _e3649_ = phi_11451_;
                    if _e3485_ {
                        phi_11466_ = true;
                    } else {
                        phi_11466_ = (1f >= _e3453_);
                    }
                    let _e3653_ = phi_11466_;
                    if _e3490_ {
                        phi_11481_ = true;
                    } else {
                        phi_11481_ = (1f >= _e3458_);
                    }
                    let _e3657_ = phi_11481_;
                    let _e3659_ = (_e3462_ - select(_e3443_, 1f, _e3645_));
                    let _e3660_ = (_e3466_ - select(_e3448_, 1f, _e3649_));
                    let _e3661_ = (_e3470_ - select(_e3453_, 1f, _e3653_));
                    let _e3662_ = (_e3474_ - select(_e3458_, 1f, _e3657_));
                    if (_e3659_ != _e3659_) {
                        phi_11496_ = true;
                    } else {
                        phi_11496_ = (0f >= _e3659_);
                    }
                    let _e3666_ = phi_11496_;
                    if (_e3660_ != _e3660_) {
                        phi_11511_ = true;
                    } else {
                        phi_11511_ = (0f >= _e3660_);
                    }
                    let _e3671_ = phi_11511_;
                    if (_e3661_ != _e3661_) {
                        phi_11526_ = true;
                    } else {
                        phi_11526_ = (0f >= _e3661_);
                    }
                    let _e3676_ = phi_11526_;
                    if (_e3662_ != _e3662_) {
                        phi_11541_ = true;
                    } else {
                        phi_11541_ = (0f >= _e3662_);
                    }
                    let _e3681_ = phi_11541_;
                    let _e3685_ = bitcast<f32>((bitcast<u32>(_e3380_) & 2147483647u));
                    let _e3688_ = bitcast<f32>((bitcast<u32>(_e3385_) & 2147483647u));
                    let _e3691_ = bitcast<f32>((bitcast<u32>(_e3390_) & 2147483647u));
                    let _e3694_ = bitcast<f32>((bitcast<u32>(_e3395_) & 2147483647u));
                    let _e3697_ = bitcast<f32>((bitcast<u32>(_e3422_) & 2147483647u));
                    let _e3700_ = bitcast<f32>((bitcast<u32>(_e3427_) & 2147483647u));
                    let _e3703_ = bitcast<f32>((bitcast<u32>(_e3432_) & 2147483647u));
                    let _e3706_ = bitcast<f32>((bitcast<u32>(_e3437_) & 2147483647u));
                    if (_e3685_ != _e3685_) {
                        phi_11556_ = true;
                    } else {
                        phi_11556_ = (_e3697_ >= _e3685_);
                    }
                    let _e3710_ = phi_11556_;
                    let _e3711_ = select(_e3685_, _e3697_, _e3710_);
                    if (_e3688_ != _e3688_) {
                        phi_11571_ = true;
                    } else {
                        phi_11571_ = (_e3700_ >= _e3688_);
                    }
                    let _e3715_ = phi_11571_;
                    let _e3716_ = select(_e3688_, _e3700_, _e3715_);
                    if (_e3691_ != _e3691_) {
                        phi_11586_ = true;
                    } else {
                        phi_11586_ = (_e3703_ >= _e3691_);
                    }
                    let _e3720_ = phi_11586_;
                    let _e3721_ = select(_e3691_, _e3703_, _e3720_);
                    if (_e3694_ != _e3694_) {
                        phi_11601_ = true;
                    } else {
                        phi_11601_ = (_e3706_ >= _e3694_);
                    }
                    let _e3725_ = phi_11601_;
                    let _e3726_ = select(_e3694_, _e3706_, _e3725_);
                    let _e3743_ = (_e3711_ <= 0f);
                    let _e3744_ = select((_e3559_ * (1.5f - (0.5f * (_e3571_ + _e3575_)))), 0f, _e3743_);
                    let _e3745_ = (_e3716_ <= 0f);
                    let _e3746_ = select((_e3560_ * (1.5f - (0.5f * (_e3572_ + _e3576_)))), 0f, _e3745_);
                    let _e3747_ = (_e3721_ <= 0f);
                    let _e3748_ = select((_e3561_ * (1.5f - (0.5f * (_e3573_ + _e3577_)))), 0f, _e3747_);
                    let _e3749_ = (_e3726_ <= 0f);
                    let _e3750_ = select((_e3562_ * (1.5f - (0.5f * (_e3574_ + _e3578_)))), 0f, _e3749_);
                    let _e3792_ = select(((_e3571_ + (0.33333334f * _e3575_)) - (((_e3571_ * _e3571_) + ((2f * _e3571_) * _e3575_)) + (0.2f * (_e3575_ * _e3575_)))), 0f, _e3743_);
                    let _e3793_ = select(((_e3572_ + (0.33333334f * _e3576_)) - (((_e3572_ * _e3572_) + ((2f * _e3572_) * _e3576_)) + (0.2f * (_e3576_ * _e3576_)))), 0f, _e3745_);
                    let _e3794_ = select(((_e3573_ + (0.33333334f * _e3577_)) - (((_e3573_ * _e3573_) + ((2f * _e3573_) * _e3577_)) + (0.2f * (_e3577_ * _e3577_)))), 0f, _e3747_);
                    let _e3795_ = select(((_e3574_ + (0.33333334f * _e3578_)) - (((_e3574_ * _e3574_) + ((2f * _e3574_) * _e3578_)) + (0.2f * (_e3578_ * _e3578_)))), 0f, _e3749_);
                    if (_e3711_ <= 1f) {
                        if (_e3716_ <= 1f) {
                            let _e3799_ = (_e3721_ <= 1f);
                            if _e3799_ {
                                phi_7233_ = (_e3726_ <= 1f);
                            } else {
                                phi_7233_ = bool();
                            }
                            let _e3802_ = phi_7233_;
                            phi_7235_ = _e3802_;
                            phi_7236_ = select(true, false, _e3799_);
                        } else {
                            phi_7235_ = bool();
                            phi_7236_ = true;
                        }
                        let _e3805_ = phi_7235_;
                        let _e3807_ = phi_7236_;
                        phi_7237_ = _e3805_;
                        phi_7238_ = _e3807_;
                    } else {
                        phi_7237_ = bool();
                        phi_7238_ = true;
                    }
                    let _e3809_ = phi_7237_;
                    let _e3811_ = phi_7238_;
                    if select(_e3809_, false, _e3811_) {
                        phi_7616_ = type_27_(type_22_(array<f32, 4>(_e3744_, _e3746_, _e3748_, _e3750_)), type_22_(array<f32, 4>(_e3792_, _e3793_, _e3794_, _e3795_)), type_22_(_e3438_));
                    } else {
                        if (_e3599_ != _e3599_) {
                            phi_11616_ = true;
                        } else {
                            phi_11616_ = (0.000001f >= _e3599_);
                        }
                        let _e3816_ = phi_11616_;
                        let _e3817_ = select(_e3599_, 0.000001f, _e3816_);
                        if (_e3600_ != _e3600_) {
                            phi_11631_ = true;
                        } else {
                            phi_11631_ = (0.000001f >= _e3600_);
                        }
                        let _e3821_ = phi_11631_;
                        let _e3822_ = select(_e3600_, 0.000001f, _e3821_);
                        if (_e3601_ != _e3601_) {
                            phi_11646_ = true;
                        } else {
                            phi_11646_ = (0.000001f >= _e3601_);
                        }
                        let _e3826_ = phi_11646_;
                        let _e3827_ = select(_e3601_, 0.000001f, _e3826_);
                        if (_e3602_ != _e3602_) {
                            phi_11661_ = true;
                        } else {
                            phi_11661_ = (0.000001f >= _e3602_);
                        }
                        let _e3831_ = phi_11661_;
                        let _e3832_ = select(_e3602_, 0.000001f, _e3831_);
                        let _e3835_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3817_)));
                        let _e3838_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3822_)));
                        let _e3841_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3827_)));
                        let _e3844_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3832_)));
                        let _e3854_ = (_e3835_ + (_e3835_ * (1f - (_e3817_ * _e3835_))));
                        let _e3856_ = (_e3838_ + (_e3838_ * (1f - (_e3822_ * _e3838_))));
                        let _e3858_ = (_e3841_ + (_e3841_ * (1f - (_e3827_ * _e3841_))));
                        let _e3860_ = (_e3844_ + (_e3844_ * (1f - (_e3832_ * _e3844_))));
                        let _e3870_ = (_e3854_ + (_e3854_ * (1f - (_e3817_ * _e3854_))));
                        let _e3872_ = (_e3856_ + (_e3856_ * (1f - (_e3822_ * _e3856_))));
                        let _e3874_ = (_e3858_ + (_e3858_ * (1f - (_e3827_ * _e3858_))));
                        let _e3876_ = (_e3860_ + (_e3860_ * (1f - (_e3832_ * _e3860_))));
                        let _e3929_ = (select((select(0f, 1f, (-1f <= _e3443_)) * select(0f, 1f, (_e3462_ <= 1f))), 1f, (_e3599_ <= 0.000001f)) <= 0f);
                        let _e3931_ = (select((select(0f, 1f, (-1f <= _e3448_)) * select(0f, 1f, (_e3466_ <= 1f))), 1f, (_e3600_ <= 0.000001f)) <= 0f);
                        let _e3933_ = (select((select(0f, 1f, (-1f <= _e3453_)) * select(0f, 1f, (_e3470_ <= 1f))), 1f, (_e3601_ <= 0.000001f)) <= 0f);
                        let _e3935_ = (select((select(0f, 1f, (-1f <= _e3458_)) * select(0f, 1f, (_e3474_ <= 1f))), 1f, (_e3602_ <= 0.000001f)) <= 0f);
                        let _e3941_ = (1f <= _e3443_);
                        let _e3943_ = (1f <= _e3448_);
                        let _e3945_ = (1f <= _e3453_);
                        let _e3947_ = (1f <= _e3458_);
                        let _e3949_ = (_e3462_ <= -1f);
                        let _e3951_ = (_e3466_ <= -1f);
                        let _e3953_ = (_e3470_ <= -1f);
                        let _e3955_ = (_e3474_ <= -1f);
                        phi_7616_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3744_, (((select(_e3659_, 0f, _e3666_) - select(_e3619_, 0f, _e3626_)) + (_e3583_ * _e3744_)) * _e3870_), _e3929_), 1f, _e3941_), -1f, _e3949_), select(select(select(_e3746_, (((select(_e3660_, 0f, _e3671_) - select(_e3620_, 0f, _e3631_)) + (_e3588_ * _e3746_)) * _e3872_), _e3931_), 1f, _e3943_), -1f, _e3951_), select(select(select(_e3748_, (((select(_e3661_, 0f, _e3676_) - select(_e3621_, 0f, _e3636_)) + (_e3593_ * _e3748_)) * _e3874_), _e3933_), 1f, _e3945_), -1f, _e3953_), select(select(select(_e3750_, (((select(_e3662_, 0f, _e3681_) - select(_e3622_, 0f, _e3641_)) + (_e3598_ * _e3750_)) * _e3876_), _e3935_), 1f, _e3947_), -1f, _e3955_))), type_22_(array<f32, 4>(select(select(select(_e3792_, ((_e3583_ * _e3792_) * _e3870_), _e3929_), 0f, _e3941_), 0f, _e3949_), select(select(select(_e3793_, ((_e3588_ * _e3793_) * _e3872_), _e3931_), 0f, _e3943_), 0f, _e3951_), select(select(select(_e3794_, ((_e3593_ * _e3794_) * _e3874_), _e3933_), 0f, _e3945_), 0f, _e3953_), select(select(select(_e3795_, ((_e3598_ * _e3795_) * _e3876_), _e3935_), 0f, _e3947_), 0f, _e3955_))), type_22_(_e3438_));
                    }
                    let _e3976_ = phi_7616_;
                    if _e160_ {
                        phi_7625_ = (_e203_ + _e3976_.member.member[0]);
                        phi_7626_ = (_e205_ + _e3351_.member.member[0]);
                    } else {
                        phi_7625_ = _e203_;
                        phi_7626_ = _e205_;
                    }
                    let _e3984_ = phi_7625_;
                    let _e3986_ = phi_7626_;
                    workgroupBarrier();
                    phi_12423_ = _e2655_;
                    phi_308_ = _e3984_;
                    phi_310_ = _e3986_;
                } else {
                    phi_12423_ = _e199_;
                    phi_308_ = f32();
                    phi_310_ = f32();
                }
                let _e3988_ = phi_12423_;
                let _e3990_ = phi_308_;
                let _e3992_ = phi_310_;
                phi_11982_ = _e3988_;
                phi_304_ = _e218_;
                phi_307_ = _e3990_;
                phi_309_ = _e3992_;
                phi_12422_ = _e3988_;
                if !(_e224_) {
                    break;
                }
            }
            let _e3995_ = phi_12422_;
            if _e3995_ {
                break;
            }
            if _e160_ {
                if (_e119_ == 0u) {
                    phi_7640_ = 0u;
                } else {
                    let _e3997_ = (_e119_ - 1u);
                    phi_7640_ = select(_e3997_, _e162_, (_e162_ < _e3997_));
                }
                let _e4001_ = phi_7640_;
                if (_e4001_ < _e119_) {
                } else {
                    break;
                }
                let _e4006_ = target_audio.member[_e4001_].x;
                let _e4010_ = target_audio.member[_e4001_].y;
                let _e4011_ = bitcast<u32>(_e4006_);
                let _e4012_ = (_e4011_ & 2139095040u);
                if (_e4012_ == 2139095040u) {
                    phi_7663_ = true;
                } else {
                    phi_7663_ = (_e4012_ == 0u);
                }
                let _e4016_ = phi_7663_;
                let _e4019_ = bitcast<u32>(_e4010_);
                let _e4020_ = (_e4019_ & 2139095040u);
                if (_e4020_ == 2139095040u) {
                    phi_7679_ = true;
                } else {
                    phi_7679_ = (_e4020_ == 0u);
                }
                let _e4024_ = phi_7679_;
                phi_7687_ = vec2<f32>(bitcast<f32>(select(_e4011_, 0u, _e4016_)), bitcast<f32>(select(_e4019_, 0u, _e4024_)));
            } else {
                phi_7687_ = vec2<f32>(0f, 0f);
            }
            let _e4029_ = phi_7687_;
            if _e160_ {
                let _e4032_ = local;
                phi_7693_ = (_e4032_ - _e4029_.x);
            } else {
                phi_7693_ = 0f;
            }
            let _e4035_ = phi_7693_;
            if _e160_ {
                let _e4038_ = local_1_;
                phi_7699_ = (_e4038_ - _e4029_.y);
            } else {
                phi_7699_ = 0f;
            }
            let _e4041_ = phi_7699_;
            if _e160_ {
                let _e4045_ = bitcast<u32>(((_e4035_ * _e4035_) + (_e4041_ * _e4041_)));
                let _e4046_ = (_e4045_ & 2139095040u);
                if (_e4046_ == 2139095040u) {
                    phi_7716_ = true;
                } else {
                    phi_7716_ = (_e4046_ == 0u);
                }
                let _e4050_ = phi_7716_;
                phi_7722_ = bitcast<f32>(select(_e4045_, 0u, _e4050_));
            } else {
                phi_7722_ = 0f;
            }
            let _e4054_ = phi_7722_;
            if _e160_ {
                let _e4060_ = bitcast<u32>(((_e4029_.x * _e4029_.x) + (_e4029_.y * _e4029_.y)));
                let _e4061_ = (_e4060_ & 2139095040u);
                if (_e4061_ == 2139095040u) {
                    phi_7743_ = true;
                } else {
                    phi_7743_ = (_e4061_ == 0u);
                }
                let _e4065_ = phi_7743_;
                phi_7749_ = bitcast<f32>(select(_e4060_, 0u, _e4065_));
            } else {
                phi_7749_ = 0f;
            }
            let _e4069_ = phi_7749_;
            let _e4070_ = bitcast<u32>(_e4054_);
            let _e4071_ = (_e4070_ & 2139095040u);
            if (_e4071_ == 2139095040u) {
                phi_7760_ = true;
            } else {
                phi_7760_ = (_e4071_ == 0u);
            }
            let _e4075_ = phi_7760_;
            let _e4078_ = bitcast<u32>(_e4069_);
            let _e4079_ = (_e4078_ & 2139095040u);
            if (_e4079_ == 2139095040u) {
                phi_7776_ = true;
            } else {
                phi_7776_ = (_e4079_ == 0u);
            }
            let _e4083_ = phi_7776_;
            let _e6199 = subgroupAdd(bitcast<f32>(select(_e4070_, 0u, _e4075_)));
            let _e6203 = subgroupAdd(bitcast<f32>(select(_e4078_, 0u, _e4083_)));
            let _e4088_ = (_e115_ == 0u);
            let _e4092_ = select(255u, _e114_, (_e114_ < 255u));
            let _e4093_ = (_e4092_ < 256u);
            if _e4093_ {
            } else {
                break;
            }
            err_scratch[_e4092_] = select(0f, _e6199, _e4088_);
            if _e4093_ {
            } else {
                break;
            }
            tgt_scratch[_e4092_] = select(0f, _e6203, _e4088_);
            workgroupBarrier();
            phi_7814_ = 128u;
            loop {
                let _e4097_ = phi_7814_;
                let _e4098_ = (_e4097_ > 0u);
                if _e4098_ {
                    if (_e114_ < _e4097_) {
                        let _e4100_ = (_e114_ + _e4097_);
                        let _e4101_ = (_e4100_ < 256u);
                        if _e4101_ {
                        } else {
                            phi_12830_ = true;
                            break;
                        }
                        let _e4103_ = err_scratch[_e4100_];
                        let _e4104_ = (_e114_ < 256u);
                        if _e4104_ {
                        } else {
                            phi_12830_ = true;
                            break;
                        }
                        let _e4106_ = err_scratch[_e114_];
                        err_scratch[_e114_] = (_e4106_ + _e4103_);
                        if _e4101_ {
                        } else {
                            phi_12830_ = true;
                            break;
                        }
                        let _e4109_ = tgt_scratch[_e4100_];
                        if _e4104_ {
                        } else {
                            phi_12830_ = true;
                            break;
                        }
                        let _e4111_ = tgt_scratch[_e114_];
                        tgt_scratch[_e114_] = (_e4111_ + _e4109_);
                    }
                    workgroupBarrier();
                    phi_7815_ = (_e4097_ >> bitcast<u32>(1i));
                } else {
                    phi_7815_ = u32();
                }
                let _e4116_ = phi_7815_;
                phi_7814_ = _e4116_;
                phi_12830_ = _e3995_;
                if !(_e4098_) {
                    break;
                }
            }
            let _e4119_ = phi_12830_;
            if _e4119_ {
                break;
            }
            let _e4121_ = err_scratch[0];
            let _e4122_ = bitcast<u32>(_e4121_);
            let _e4123_ = (_e4122_ & 2139095040u);
            if (_e4123_ == 2139095040u) {
                phi_7875_ = true;
            } else {
                phi_7875_ = (_e4123_ == 0u);
            }
            let _e4127_ = phi_7875_;
            let _e4131_ = tgt_scratch[0];
            let _e4132_ = bitcast<u32>(_e4131_);
            let _e4133_ = (_e4132_ & 2139095040u);
            if (_e4133_ == 2139095040u) {
                phi_7893_ = true;
            } else {
                phi_7893_ = (_e4133_ == 0u);
            }
            let _e4137_ = phi_7893_;
            workgroupBarrier();
            if (_e114_ == 0u) {
                if (_e113_.x < _e131_) {
                    if (_e113_.y < _e151_) {
                        if (_e113_.z < select(1u, _e142_, (1u < _e142_))) {
                            let _e4146_ = uniforms.member.member_5_;
                            if ((bitcast<u32>(_e4146_) & 2139095040u) != 2139095040u) {
                                let _e4150_ = (_e4146_ > 0.000000000001f);
                                phi_7933_ = select(f32(), _e4146_, _e4150_);
                                phi_7934_ = select(true, false, _e4150_);
                            } else {
                                phi_7933_ = f32();
                                phi_7934_ = true;
                            }
                            let _e4154_ = phi_7933_;
                            let _e4156_ = phi_7934_;
                            let _e4158_ = (bitcast<f32>(select(_e4132_, 0u, _e4137_)) + select(_e4154_, 0.000000000001f, _e4156_));
                            if ((bitcast<u32>(_e4158_) & 2139095040u) != 2139095040u) {
                                let _e4162_ = (_e4158_ <= 0f);
                                if _e4162_ {
                                    phi_8032_ = f32();
                                } else {
                                    if (_e4158_ < 0.00001f) {
                                        let _e4194_ = (select(0.000000000000000000000001f, _e4158_, (_e4158_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                        if (_e4194_ != _e4194_) {
                                            phi_11736_ = true;
                                        } else {
                                            phi_11736_ = (0.000001f >= _e4194_);
                                        }
                                        let _e4198_ = phi_11736_;
                                        let _e4199_ = select(_e4194_, 0.000001f, _e4198_);
                                        let _e4202_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4199_)));
                                        let _e4206_ = (_e4202_ + (_e4202_ * (1f - (_e4199_ * _e4202_))));
                                        let _e4210_ = (_e4206_ + (_e4206_ * (1f - (_e4199_ * _e4206_))));
                                        let _e4216_ = bitcast<u32>(((_e4210_ + (_e4210_ * (1f - (_e4199_ * _e4210_)))) * 18446744000000000000f));
                                        let _e4217_ = (_e4216_ & 2139095040u);
                                        if (_e4217_ == 2139095040u) {
                                            phi_7990_ = true;
                                        } else {
                                            phi_7990_ = (_e4217_ == 0u);
                                        }
                                        let _e4221_ = phi_7990_;
                                        phi_8031_ = bitcast<f32>(select(_e4216_, 0u, _e4221_));
                                    } else {
                                        if (_e4158_ != _e4158_) {
                                            phi_11676_ = true;
                                        } else {
                                            phi_11676_ = (0.000001f >= _e4158_);
                                        }
                                        let _e4167_ = phi_11676_;
                                        let _e4168_ = select(_e4158_, 0.000001f, _e4167_);
                                        let _e4171_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4168_)));
                                        let _e4175_ = (_e4171_ + (_e4171_ * (1f - (_e4168_ * _e4171_))));
                                        let _e4179_ = (_e4175_ + (_e4175_ * (1f - (_e4168_ * _e4175_))));
                                        let _e4184_ = bitcast<u32>((_e4179_ + (_e4179_ * (1f - (_e4168_ * _e4179_)))));
                                        let _e4185_ = (_e4184_ & 2139095040u);
                                        if (_e4185_ == 2139095040u) {
                                            phi_8025_ = true;
                                        } else {
                                            phi_8025_ = (_e4185_ == 0u);
                                        }
                                        let _e4189_ = phi_8025_;
                                        phi_8031_ = bitcast<f32>(select(_e4184_, 0u, _e4189_));
                                    }
                                    let _e4225_ = phi_8031_;
                                    phi_8032_ = _e4225_;
                                }
                                let _e4227_ = phi_8032_;
                                phi_8033_ = _e4227_;
                                phi_8034_ = _e4162_;
                            } else {
                                phi_8033_ = f32();
                                phi_8034_ = true;
                            }
                            let _e4229_ = phi_8033_;
                            let _e4231_ = phi_8034_;
                            let _e4234_ = bitcast<u32>((bitcast<f32>(select(_e4122_, 0u, _e4127_)) * select(_e4229_, 0f, _e4231_)));
                            let _e4235_ = (_e4234_ & 2139095040u);
                            if (_e4235_ == 2139095040u) {
                                phi_8050_ = true;
                            } else {
                                phi_8050_ = (_e4235_ == 0u);
                            }
                            let _e4239_ = phi_8050_;
                            let _e4245_ = ((((_e113_.z * _e131_) + _e113_.x) * _e151_) + _e113_.y);
                            if (_e123_ == 0u) {
                                phi_8070_ = 0u;
                            } else {
                                let _e4247_ = (_e123_ - 1u);
                                phi_8070_ = select(_e4247_, _e4245_, (_e4245_ < _e4247_));
                            }
                            let _e4251_ = phi_8070_;
                            if (_e4251_ < _e123_) {
                            } else {
                                break;
                            }
                            seed_losses.member[_e4251_] = bitcast<f32>(select(_e4234_, 0u, _e4239_));
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
