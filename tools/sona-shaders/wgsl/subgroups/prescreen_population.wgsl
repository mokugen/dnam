struct type_6_ {
    member: array<f32>,
}

struct type_15_ {
    member: u32,
    member_1_: u32,
}

struct type_21_ {
    member: f32,
    member_1_: f32,
}

struct type_22_ {
    member: f32,
    member_1_: f32,
    member_2_: f32,
    member_3_: f32,
    member_4_: f32,
    member_5_: f32,
}

struct type_23_ {
    member: array<f32, 4>,
}

struct type_25_ {
    member: array<vec2<f32>>,
}

struct type_27_ {
    member: u32,
    member_1_: u32,
    member_2_: u32,
    member_3_: u32,
    member_4_: f32,
    member_5_: f32,
    member_6_: u32,
    member_7_: u32,
}

struct type_28_ {
    member: type_27_,
}

var<private> wg_id_1_: vec3<u32>;
var<private> lid_1_: u32;
var<private> sg_lid_1_: u32;
var<workgroup> scan_scratch: array<f32, 1536>;
@group(0) @binding(2) 
var<storage> target_audio: type_25_;
@group(0) @binding(0) 
var<uniform> uniforms: type_28_;
@group(0) @binding(1) 
var<storage> input_audio: type_25_;
@group(0) @binding(3) 
var<storage> seed_params: type_6_;
@group(0) @binding(4) 
var<storage, read_write> seed_losses: type_6_;
var<workgroup> err_scratch: array<f32, 256>;
var<workgroup> tgt_scratch: array<f32, 256>;

fn function_() {
    var phi_221_: u32;
    var phi_228_: u32;
    var phi_251_: u32;
    var phi_274_: bool;
    var phi_290_: bool;
    var phi_298_: vec2<f32>;
    var phi_10393_: bool;
    var phi_301_: type_15_;
    var phi_304_: f32;
    var phi_306_: f32;
    var phi_302_: type_15_;
    var phi_329_: type_15_;
    var phi_347_: u32;
    var phi_358_: u32;
    var phi_370_: u32;
    var phi_389_: u32;
    var phi_422_: f32;
    var phi_428_: f32;
    var phi_7237_: bool;
    var phi_599_: bool;
    var phi_605_: f32;
    var phi_606_: f32;
    var phi_525_: f32;
    var phi_527_: f32;
    var phi_528_: bool;
    var phi_548_: f32;
    var phi_607_: f32;
    var phi_608_: f32;
    var phi_7297_: bool;
    var phi_694_: bool;
    var phi_7357_: bool;
    var phi_659_: bool;
    var phi_700_: f32;
    var phi_701_: f32;
    var phi_702_: f32;
    var phi_703_: bool;
    var phi_719_: u32;
    var phi_737_: bool;
    var phi_754_: u32;
    var phi_772_: bool;
    var phi_789_: u32;
    var phi_807_: bool;
    var phi_824_: u32;
    var phi_842_: bool;
    var phi_859_: u32;
    var phi_877_: bool;
    var phi_894_: u32;
    var phi_912_: bool;
    var phi_7417_: bool;
    var phi_7477_: bool;
    var phi_1493_: bool;
    var phi_1499_: f32;
    var phi_1500_: f32;
    var phi_1419_: f32;
    var phi_1421_: f32;
    var phi_1422_: bool;
    var phi_1501_: f32;
    var phi_1513_: u32;
    var phi_1531_: bool;
    var phi_1548_: u32;
    var phi_1566_: bool;
    var phi_1583_: u32;
    var phi_1601_: bool;
    var phi_1618_: u32;
    var phi_1636_: bool;
    var phi_1653_: u32;
    var phi_1777_: bool;
    var phi_1783_: f32;
    var phi_1784_: f32;
    var phi_1703_: f32;
    var phi_1705_: f32;
    var phi_1706_: bool;
    var phi_1726_: f32;
    var phi_1785_: f32;
    var phi_1797_: u32;
    var phi_1921_: bool;
    var phi_1927_: f32;
    var phi_1928_: f32;
    var phi_1847_: f32;
    var phi_1849_: f32;
    var phi_1850_: bool;
    var phi_1870_: f32;
    var phi_1929_: f32;
    var phi_1941_: u32;
    var phi_2065_: bool;
    var phi_2071_: f32;
    var phi_2072_: f32;
    var phi_1991_: f32;
    var phi_1993_: f32;
    var phi_1994_: bool;
    var phi_2014_: f32;
    var phi_2073_: f32;
    var phi_2085_: u32;
    var phi_2209_: bool;
    var phi_2215_: f32;
    var phi_2216_: f32;
    var phi_2135_: f32;
    var phi_2137_: f32;
    var phi_2138_: bool;
    var phi_2158_: f32;
    var phi_2217_: f32;
    var phi_2239_: type_22_;
    var phi_2299_: u32;
    var phi_2300_: u32;
    var phi_10197_: bool;
    var phi_2562_: u32;
    var phi_2563_: u32;
    var phi_10498_: bool;
    var phi_3105_: bool;
    var phi_3121_: bool;
    var phi_3129_: type_21_;
    var phi_3152_: type_22_;
    var phi_3212_: u32;
    var phi_3213_: u32;
    var phi_10513_: bool;
    var phi_3475_: u32;
    var phi_3476_: u32;
    var phi_10533_: bool;
    var phi_4018_: bool;
    var phi_4034_: bool;
    var phi_4042_: type_21_;
    var phi_4107_: type_15_;
    var phi_4110_: type_23_;
    var phi_4112_: type_23_;
    var phi_4108_: type_15_;
    var phi_4135_: type_15_;
    var phi_4188_: bool;
    var phi_4204_: bool;
    var phi_4220_: bool;
    var phi_4236_: bool;
    var phi_7537_: bool;
    var phi_7552_: bool;
    var phi_7567_: bool;
    var phi_7582_: bool;
    var phi_7597_: bool;
    var phi_7612_: bool;
    var phi_7627_: bool;
    var phi_7642_: bool;
    var phi_7657_: bool;
    var phi_7672_: bool;
    var phi_7687_: bool;
    var phi_7702_: bool;
    var phi_7717_: bool;
    var phi_7732_: bool;
    var phi_7747_: bool;
    var phi_7762_: bool;
    var phi_7777_: bool;
    var phi_7792_: bool;
    var phi_7807_: bool;
    var phi_7822_: bool;
    var phi_7837_: bool;
    var phi_7852_: bool;
    var phi_7867_: bool;
    var phi_7882_: bool;
    var phi_7897_: bool;
    var phi_7912_: bool;
    var phi_7927_: bool;
    var phi_7942_: bool;
    var phi_7957_: bool;
    var phi_7972_: bool;
    var phi_7987_: bool;
    var phi_8002_: bool;
    var phi_4643_: bool;
    var phi_4659_: bool;
    var phi_4675_: bool;
    var phi_4691_: bool;
    var phi_4713_: bool;
    var phi_4729_: bool;
    var phi_4745_: bool;
    var phi_4761_: bool;
    var phi_8017_: bool;
    var phi_8032_: bool;
    var phi_8047_: bool;
    var phi_8062_: bool;
    var phi_8077_: bool;
    var phi_8092_: bool;
    var phi_8107_: bool;
    var phi_8122_: bool;
    var phi_8137_: bool;
    var phi_8152_: bool;
    var phi_8167_: bool;
    var phi_8182_: bool;
    var phi_8197_: bool;
    var phi_8212_: bool;
    var phi_8227_: bool;
    var phi_8242_: bool;
    var phi_8257_: bool;
    var phi_8272_: bool;
    var phi_8287_: bool;
    var phi_8302_: bool;
    var phi_8317_: bool;
    var phi_8332_: bool;
    var phi_8347_: bool;
    var phi_8362_: bool;
    var phi_8377_: bool;
    var phi_8392_: bool;
    var phi_8407_: bool;
    var phi_8422_: bool;
    var phi_8437_: bool;
    var phi_8452_: bool;
    var phi_8467_: bool;
    var phi_8482_: bool;
    var phi_5168_: bool;
    var phi_5184_: bool;
    var phi_5200_: bool;
    var phi_5216_: bool;
    var phi_4111_: type_23_;
    var phi_4113_: type_23_;
    var phi_10548_: bool;
    var phi_10976_: bool;
    var phi_5304_: f32;
    var phi_5316_: f32;
    var phi_8497_: bool;
    var phi_8512_: bool;
    var phi_8527_: bool;
    var phi_8587_: bool;
    var phi_5372_: f32;
    var phi_8647_: bool;
    var phi_8707_: bool;
    var phi_5422_: f32;
    var phi_5451_: f32;
    var phi_5453_: f32;
    var phi_5454_: bool;
    var phi_5475_: f32;
    var phi_5477_: f32;
    var phi_5478_: bool;
    var phi_8767_: bool;
    var phi_8827_: bool;
    var phi_8887_: bool;
    var phi_8947_: bool;
    var phi_9007_: bool;
    var phi_9067_: bool;
    var phi_9127_: bool;
    var phi_9187_: bool;
    var phi_9247_: bool;
    var phi_9427_: bool;
    var phi_9487_: bool;
    var phi_9547_: bool;
    var phi_9607_: bool;
    var phi_9667_: bool;
    var phi_9727_: bool;
    var phi_9787_: bool;
    var phi_9847_: bool;
    var phi_9907_: bool;
    var phi_6756_: f32;
    var phi_6757_: f32;
    var phi_10754_: bool;
    var phi_305_: f32;
    var phi_307_: f32;
    var phi_10753_: bool;
    var phi_6771_: u32;
    var phi_6794_: bool;
    var phi_6810_: bool;
    var phi_6818_: vec2<f32>;
    var local: f32;
    var phi_6824_: f32;
    var local_1_: f32;
    var phi_6830_: f32;
    var phi_6847_: bool;
    var phi_6853_: f32;
    var phi_6874_: bool;
    var phi_6880_: f32;
    var phi_6891_: bool;
    var phi_6907_: bool;
    var phi_6945_: u32;
    var phi_6946_: u32;
    var phi_10989_: bool;
    var phi_7006_: bool;
    var phi_7024_: bool;
    var phi_7064_: f32;
    var phi_7065_: bool;
    var phi_10087_: bool;
    var phi_7156_: bool;
    var phi_10147_: bool;
    var phi_7121_: bool;
    var phi_7162_: f32;
    var phi_7163_: f32;
    var phi_7164_: f32;
    var phi_7165_: bool;
    var phi_7181_: bool;
    var phi_7201_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e110_ = wg_id_1_;
            let _e111_ = lid_1_;
            let _e112_ = sg_lid_1_;
            let _e114_ = arrayLength((&input_audio.member));
            let _e116_ = arrayLength((&target_audio.member));
            let _e118_ = arrayLength((&seed_params.member));
            let _e120_ = arrayLength((&seed_losses.member));
            let _e126_ = uniforms.member.member;
            let _e128_ = select(1u, _e126_, (1u < _e126_));
            let _e131_ = uniforms.member.member_1_;
            let _e134_ = uniforms.member.member_2_;
            let _e136_ = select(1u, _e134_, (1u < _e134_));
            let _e139_ = uniforms.member.member_3_;
            if (_e131_ == 0u) {
                phi_221_ = 1u;
            } else {
                phi_221_ = (((_e131_ - 1u) >> bitcast<u32>(8i)) + 1u);
            }
            let _e148_ = phi_221_;
            let _e149_ = (_e110_.y * 256u);
            if (_e131_ > _e149_) {
                phi_228_ = (_e131_ - _e149_);
            } else {
                phi_228_ = 0u;
            }
            let _e153_ = phi_228_;
            let _e156_ = (_e149_ + _e111_);
            let _e157_ = (_e156_ < _e131_);
            let _e159_ = ((_e110_.z * _e131_) + _e156_);
            if _e157_ {
                if (_e114_ == 0u) {
                    phi_251_ = 0u;
                } else {
                    let _e161_ = (_e114_ - 1u);
                    phi_251_ = select(_e161_, _e159_, (_e159_ < _e161_));
                }
                let _e165_ = phi_251_;
                if (_e165_ < _e114_) {
                } else {
                    break;
                }
                let _e170_ = input_audio.member[_e165_].x;
                let _e174_ = input_audio.member[_e165_].y;
                let _e175_ = bitcast<u32>(_e170_);
                let _e176_ = (_e175_ & 2139095040u);
                if (_e176_ == 2139095040u) {
                    phi_274_ = true;
                } else {
                    phi_274_ = (_e176_ == 0u);
                }
                let _e180_ = phi_274_;
                let _e183_ = bitcast<u32>(_e174_);
                let _e184_ = (_e183_ & 2139095040u);
                if (_e184_ == 2139095040u) {
                    phi_290_ = true;
                } else {
                    phi_290_ = (_e184_ == 0u);
                }
                let _e188_ = phi_290_;
                phi_298_ = vec2<f32>(bitcast<f32>(select(_e175_, 0u, _e180_)), bitcast<f32>(select(_e183_, 0u, _e188_)));
            } else {
                phi_298_ = vec2<f32>(0f, 0f);
            }
            let _e193_ = phi_298_;
            phi_10393_ = false;
            phi_301_ = type_15_(0u, _e136_);
            phi_304_ = 0f;
            phi_306_ = 0f;
            loop {
                let _e196_ = phi_10393_;
                let _e198_ = phi_301_;
                let _e200_ = phi_304_;
                let _e202_ = phi_306_;
                local = _e202_;
                local_1_ = _e200_;
                if (_e198_.member < _e198_.member_1_) {
                    phi_302_ = type_15_((_e198_.member + 1u), _e198_.member_1_);
                    phi_329_ = type_15_(1u, _e198_.member);
                } else {
                    phi_302_ = _e198_;
                    phi_329_ = type_15_(0u, u32());
                }
                let _e215_ = phi_302_;
                let _e217_ = phi_329_;
                let _e221_ = (bitcast<i32>(_e217_.member) != 0i);
                if _e221_ {
                    if (_e128_ == 0u) {
                        phi_347_ = 0u;
                    } else {
                        let _e223_ = (_e128_ - 1u);
                        phi_347_ = select(_e223_, _e110_.x, (_e110_.x < _e223_));
                    }
                    let _e227_ = phi_347_;
                    if (_e118_ >= ((_e128_ * _e136_) * 16u)) {
                        phi_358_ = (((_e227_ * _e136_) + _e217_.member_1_) * 16u);
                    } else {
                        phi_358_ = (_e227_ * 16u);
                    }
                    let _e236_ = phi_358_;
                    let _e237_ = (_e118_ == 0u);
                    if _e237_ {
                        phi_370_ = 0u;
                    } else {
                        let _e238_ = (_e118_ - 1u);
                        phi_370_ = select(_e238_, _e236_, (_e236_ < _e238_));
                    }
                    let _e242_ = phi_370_;
                    if (_e242_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e246_ = seed_params.member[_e242_];
                    let _e247_ = (_e236_ + 1u);
                    if _e237_ {
                        phi_389_ = 0u;
                    } else {
                        let _e248_ = (_e118_ - 1u);
                        phi_389_ = select(_e248_, _e247_, (_e247_ < _e248_));
                    }
                    let _e252_ = phi_389_;
                    if (_e252_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e256_ = seed_params.member[_e252_];
                    let _e260_ = select(0.001f, _e246_, ((bitcast<u32>(_e246_) & 2139095040u) != 2139095040u));
                    if (_e260_ != _e260_) {
                        phi_422_ = 0.00001f;
                    } else {
                        let _e263_ = select(_e260_, 0.00001f, (_e260_ < 0.00001f));
                        phi_422_ = select(_e263_, 0.495f, (_e263_ > 0.495f));
                    }
                    let _e267_ = phi_422_;
                    let _e268_ = (_e267_ <= 0.25f);
                    if _e268_ {
                        phi_428_ = _e267_;
                    } else {
                        phi_428_ = (0.5f - _e267_);
                    }
                    let _e271_ = phi_428_;
                    let _e272_ = (3.1415927f * _e271_);
                    let _e273_ = (_e272_ * _e272_);
                    let _e284_ = (_e272_ * (1f + (_e273_ * (-0.16666667f + (_e273_ * (0.008333334f + (_e273_ * (-0.0001984127f + (_e273_ * (0.0000027557319f + (_e273_ * -0.000000025052108f)))))))))));
                    let _e294_ = (1f + (_e273_ * (-0.5f + (_e273_ * (0.041666668f + (_e273_ * (-0.0013888889f + (_e273_ * (0.000024801588f + (_e273_ * -0.0000002755732f))))))))));
                    let _e296_ = select(_e284_, _e294_, _e268_);
                    if (_e296_ != _e296_) {
                        phi_7237_ = true;
                    } else {
                        phi_7237_ = (0.000001f >= _e296_);
                    }
                    let _e300_ = phi_7237_;
                    let _e301_ = select(_e296_, 0.000001f, _e300_);
                    let _e304_ = bitcast<f32>((2129859011u - bitcast<u32>(_e301_)));
                    let _e308_ = (_e304_ + (_e304_ * (1f - (_e301_ * _e304_))));
                    let _e312_ = (_e308_ + (_e308_ * (1f - (_e301_ * _e308_))));
                    let _e317_ = (select(_e294_, _e284_, _e268_) * (_e312_ + (_e312_ * (1f - (_e301_ * _e312_)))));
                    let _e318_ = bitcast<u32>(_e256_);
                    let _e319_ = (_e318_ & 2139095040u);
                    if (_e319_ != 2139095040u) {
                        let _e321_ = (_e319_ == 2139095040u);
                        if _e321_ {
                            if ((_e318_ & 2147483647u) > 2139095040u) {
                                phi_527_ = f32();
                                phi_528_ = true;
                            } else {
                                let _e336_ = ((_e318_ & 2147483648u) == 0u);
                                if _e336_ {
                                    phi_525_ = 64f;
                                } else {
                                    phi_525_ = f32();
                                }
                                let _e338_ = phi_525_;
                                phi_527_ = _e338_;
                                phi_528_ = select(true, false, _e336_);
                            }
                            let _e341_ = phi_527_;
                            let _e343_ = phi_528_;
                            if _e343_ {
                                phi_548_ = 0.05f;
                            } else {
                                phi_548_ = _e341_;
                            }
                            let _e345_ = phi_548_;
                            phi_607_ = _e345_;
                        } else {
                            if (_e256_ < 0.05f) {
                                phi_606_ = 0.05f;
                            } else {
                                if (_e256_ > 64f) {
                                    phi_605_ = 64f;
                                } else {
                                    if _e321_ {
                                        phi_599_ = true;
                                    } else {
                                        phi_599_ = (_e319_ == 0u);
                                    }
                                    let _e326_ = phi_599_;
                                    phi_605_ = bitcast<f32>(select(_e318_, 0u, _e326_));
                                }
                                let _e330_ = phi_605_;
                                phi_606_ = _e330_;
                            }
                            let _e332_ = phi_606_;
                            phi_607_ = _e332_;
                        }
                        let _e347_ = phi_607_;
                        phi_608_ = _e347_;
                    } else {
                        phi_608_ = 0.70710677f;
                    }
                    let _e349_ = phi_608_;
                    if ((bitcast<u32>(_e349_) & 2139095040u) != 2139095040u) {
                        let _e353_ = (_e349_ <= 0f);
                        if _e353_ {
                            phi_701_ = f32();
                        } else {
                            if (_e349_ < 0.00001f) {
                                let _e385_ = (select(0.000000000000000000000001f, _e349_, (_e349_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e385_ != _e385_) {
                                    phi_7357_ = true;
                                } else {
                                    phi_7357_ = (0.000001f >= _e385_);
                                }
                                let _e389_ = phi_7357_;
                                let _e390_ = select(_e385_, 0.000001f, _e389_);
                                let _e393_ = bitcast<f32>((2129859011u - bitcast<u32>(_e390_)));
                                let _e397_ = (_e393_ + (_e393_ * (1f - (_e390_ * _e393_))));
                                let _e401_ = (_e397_ + (_e397_ * (1f - (_e390_ * _e397_))));
                                let _e407_ = bitcast<u32>(((_e401_ + (_e401_ * (1f - (_e390_ * _e401_)))) * 18446744000000000000f));
                                let _e408_ = (_e407_ & 2139095040u);
                                if (_e408_ == 2139095040u) {
                                    phi_659_ = true;
                                } else {
                                    phi_659_ = (_e408_ == 0u);
                                }
                                let _e412_ = phi_659_;
                                phi_700_ = bitcast<f32>(select(_e407_, 0u, _e412_));
                            } else {
                                if (_e349_ != _e349_) {
                                    phi_7297_ = true;
                                } else {
                                    phi_7297_ = (0.000001f >= _e349_);
                                }
                                let _e358_ = phi_7297_;
                                let _e359_ = select(_e349_, 0.000001f, _e358_);
                                let _e362_ = bitcast<f32>((2129859011u - bitcast<u32>(_e359_)));
                                let _e366_ = (_e362_ + (_e362_ * (1f - (_e359_ * _e362_))));
                                let _e370_ = (_e366_ + (_e366_ * (1f - (_e359_ * _e366_))));
                                let _e375_ = bitcast<u32>((_e370_ + (_e370_ * (1f - (_e359_ * _e370_)))));
                                let _e376_ = (_e375_ & 2139095040u);
                                if (_e376_ == 2139095040u) {
                                    phi_694_ = true;
                                } else {
                                    phi_694_ = (_e376_ == 0u);
                                }
                                let _e380_ = phi_694_;
                                phi_700_ = bitcast<f32>(select(_e375_, 0u, _e380_));
                            }
                            let _e416_ = phi_700_;
                            phi_701_ = _e416_;
                        }
                        let _e418_ = phi_701_;
                        phi_702_ = _e418_;
                        phi_703_ = _e353_;
                    } else {
                        phi_702_ = f32();
                        phi_703_ = true;
                    }
                    let _e420_ = phi_702_;
                    let _e422_ = phi_703_;
                    let _e423_ = select(_e420_, 0f, _e422_);
                    let _e424_ = (_e236_ + 6u);
                    if _e237_ {
                        phi_719_ = 0u;
                    } else {
                        let _e425_ = (_e118_ - 1u);
                        phi_719_ = select(_e425_, _e424_, (_e424_ < _e425_));
                    }
                    let _e429_ = phi_719_;
                    if (_e429_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e433_ = seed_params.member[_e429_];
                    let _e434_ = bitcast<u32>(_e433_);
                    let _e435_ = (_e434_ & 2139095040u);
                    if (_e435_ == 2139095040u) {
                        phi_737_ = true;
                    } else {
                        phi_737_ = (_e435_ == 0u);
                    }
                    let _e439_ = phi_737_;
                    let _e441_ = bitcast<f32>(select(_e434_, 0u, _e439_));
                    let _e442_ = (_e236_ + 7u);
                    if _e237_ {
                        phi_754_ = 0u;
                    } else {
                        let _e443_ = (_e118_ - 1u);
                        phi_754_ = select(_e443_, _e442_, (_e442_ < _e443_));
                    }
                    let _e447_ = phi_754_;
                    if (_e447_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e451_ = seed_params.member[_e447_];
                    let _e452_ = bitcast<u32>(_e451_);
                    let _e453_ = (_e452_ & 2139095040u);
                    if (_e453_ == 2139095040u) {
                        phi_772_ = true;
                    } else {
                        phi_772_ = (_e453_ == 0u);
                    }
                    let _e457_ = phi_772_;
                    let _e459_ = bitcast<f32>(select(_e452_, 0u, _e457_));
                    let _e460_ = (_e236_ + 8u);
                    if _e237_ {
                        phi_789_ = 0u;
                    } else {
                        let _e461_ = (_e118_ - 1u);
                        phi_789_ = select(_e461_, _e460_, (_e460_ < _e461_));
                    }
                    let _e465_ = phi_789_;
                    if (_e465_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e469_ = seed_params.member[_e465_];
                    let _e470_ = bitcast<u32>(_e469_);
                    let _e471_ = (_e470_ & 2139095040u);
                    if (_e471_ == 2139095040u) {
                        phi_807_ = true;
                    } else {
                        phi_807_ = (_e471_ == 0u);
                    }
                    let _e475_ = phi_807_;
                    let _e477_ = bitcast<f32>(select(_e470_, 0u, _e475_));
                    let _e478_ = (_e236_ + 9u);
                    if _e237_ {
                        phi_824_ = 0u;
                    } else {
                        let _e479_ = (_e118_ - 1u);
                        phi_824_ = select(_e479_, _e478_, (_e478_ < _e479_));
                    }
                    let _e483_ = phi_824_;
                    if (_e483_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e487_ = seed_params.member[_e483_];
                    let _e488_ = bitcast<u32>(_e487_);
                    let _e489_ = (_e488_ & 2139095040u);
                    if (_e489_ == 2139095040u) {
                        phi_842_ = true;
                    } else {
                        phi_842_ = (_e489_ == 0u);
                    }
                    let _e493_ = phi_842_;
                    let _e495_ = bitcast<f32>(select(_e488_, 0u, _e493_));
                    let _e496_ = (_e236_ + 10u);
                    if _e237_ {
                        phi_859_ = 0u;
                    } else {
                        let _e497_ = (_e118_ - 1u);
                        phi_859_ = select(_e497_, _e496_, (_e496_ < _e497_));
                    }
                    let _e501_ = phi_859_;
                    if (_e501_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e505_ = seed_params.member[_e501_];
                    let _e506_ = bitcast<u32>(_e505_);
                    let _e507_ = (_e506_ & 2139095040u);
                    if (_e507_ == 2139095040u) {
                        phi_877_ = true;
                    } else {
                        phi_877_ = (_e507_ == 0u);
                    }
                    let _e511_ = phi_877_;
                    let _e513_ = bitcast<f32>(select(_e506_, 0u, _e511_));
                    let _e514_ = (_e236_ + 11u);
                    if _e237_ {
                        phi_894_ = 0u;
                    } else {
                        let _e515_ = (_e118_ - 1u);
                        phi_894_ = select(_e515_, _e514_, (_e514_ < _e515_));
                    }
                    let _e519_ = phi_894_;
                    if (_e519_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e523_ = seed_params.member[_e519_];
                    let _e524_ = bitcast<u32>(_e523_);
                    let _e525_ = (_e524_ & 2139095040u);
                    if (_e525_ == 2139095040u) {
                        phi_912_ = true;
                    } else {
                        phi_912_ = (_e525_ == 0u);
                    }
                    let _e529_ = phi_912_;
                    let _e531_ = bitcast<f32>(select(_e524_, 0u, _e529_));
                    let _e537_ = select(select(select(0f, 0.999995f, (0.999995f <= _e441_)), _e441_, (_e441_ <= 0.999995f)), 0f, (_e441_ <= 0f));
                    let _e543_ = select(select(select(0f, 1f, (1f <= _e459_)), _e459_, (_e459_ <= 1f)), 0f, (_e459_ <= 0f));
                    let _e555_ = select(select(select(0f, 64f, (64f <= _e495_)), _e495_, (_e495_ <= 64f)), 0f, (_e495_ <= 0f));
                    let _e561_ = select(select(select(0f, 1f, (1f <= _e513_)), _e513_, (_e513_ <= 1f)), 0f, (_e513_ <= 0f));
                    let _e567_ = select(select(select(0f, 1f, (1f <= _e531_)), _e531_, (_e531_ <= 1f)), 0f, (_e531_ <= 0f));
                    if (_e317_ != _e317_) {
                        phi_7417_ = true;
                    } else {
                        phi_7417_ = (0.00001f >= _e317_);
                    }
                    let _e571_ = phi_7417_;
                    let _e572_ = select(_e317_, 0.00001f, _e571_);
                    if (_e423_ != _e423_) {
                        phi_7477_ = true;
                    } else {
                        phi_7477_ = (0f >= _e423_);
                    }
                    let _e576_ = phi_7477_;
                    let _e577_ = select(_e423_, 0f, _e576_);
                    let _e580_ = (1f + (_e572_ * (_e572_ + _e577_)));
                    let _e583_ = bitcast<f32>((2129859011u - bitcast<u32>(_e580_)));
                    let _e587_ = (_e583_ + (_e583_ * (1f - (_e580_ * _e583_))));
                    let _e591_ = (_e587_ + (_e587_ * (1f - (_e580_ * _e587_))));
                    let _e592_ = (_e572_ * _e591_);
                    let _e593_ = (_e572_ * _e592_);
                    let _e594_ = (_e317_ + _e317_);
                    let _e595_ = bitcast<u32>(_e594_);
                    let _e596_ = (_e595_ & 2139095040u);
                    let _e597_ = (_e596_ == 2139095040u);
                    if _e597_ {
                        if ((_e595_ & 2147483647u) > 2139095040u) {
                            phi_1421_ = f32();
                            phi_1422_ = true;
                        } else {
                            let _e612_ = ((_e595_ & 2147483648u) == 0u);
                            if _e612_ {
                                phi_1419_ = 128f;
                            } else {
                                phi_1419_ = f32();
                            }
                            let _e614_ = phi_1419_;
                            phi_1421_ = _e614_;
                            phi_1422_ = select(true, false, _e612_);
                        }
                        let _e617_ = phi_1421_;
                        let _e619_ = phi_1422_;
                        phi_1501_ = select(_e617_, 0f, _e619_);
                    } else {
                        if (_e594_ < 0f) {
                            phi_1500_ = 0f;
                        } else {
                            if (_e594_ > 128f) {
                                phi_1499_ = 128f;
                            } else {
                                if _e597_ {
                                    phi_1493_ = true;
                                } else {
                                    phi_1493_ = (_e596_ == 0u);
                                }
                                let _e602_ = phi_1493_;
                                phi_1499_ = bitcast<f32>(select(_e595_, 0u, _e602_));
                            }
                            let _e606_ = phi_1499_;
                            phi_1500_ = _e606_;
                        }
                        let _e608_ = phi_1500_;
                        phi_1501_ = _e608_;
                    }
                    let _e622_ = phi_1501_;
                    let _e623_ = (_e236_ + 2u);
                    if _e237_ {
                        phi_1513_ = 0u;
                    } else {
                        let _e624_ = (_e118_ - 1u);
                        phi_1513_ = select(_e624_, _e623_, (_e623_ < _e624_));
                    }
                    let _e628_ = phi_1513_;
                    if (_e628_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e632_ = seed_params.member[_e628_];
                    let _e633_ = bitcast<u32>(_e632_);
                    let _e634_ = (_e633_ & 2139095040u);
                    if (_e634_ == 2139095040u) {
                        phi_1531_ = true;
                    } else {
                        phi_1531_ = (_e634_ == 0u);
                    }
                    let _e638_ = phi_1531_;
                    let _e640_ = bitcast<f32>(select(_e633_, 0u, _e638_));
                    let _e641_ = (_e236_ + 3u);
                    if _e237_ {
                        phi_1548_ = 0u;
                    } else {
                        let _e642_ = (_e118_ - 1u);
                        phi_1548_ = select(_e642_, _e641_, (_e641_ < _e642_));
                    }
                    let _e646_ = phi_1548_;
                    if (_e646_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e650_ = seed_params.member[_e646_];
                    let _e651_ = bitcast<u32>(_e650_);
                    let _e652_ = (_e651_ & 2139095040u);
                    if (_e652_ == 2139095040u) {
                        phi_1566_ = true;
                    } else {
                        phi_1566_ = (_e652_ == 0u);
                    }
                    let _e656_ = phi_1566_;
                    let _e658_ = bitcast<f32>(select(_e651_, 0u, _e656_));
                    let _e659_ = (_e236_ + 4u);
                    if _e237_ {
                        phi_1583_ = 0u;
                    } else {
                        let _e660_ = (_e118_ - 1u);
                        phi_1583_ = select(_e660_, _e659_, (_e659_ < _e660_));
                    }
                    let _e664_ = phi_1583_;
                    if (_e664_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e668_ = seed_params.member[_e664_];
                    let _e669_ = bitcast<u32>(_e668_);
                    let _e670_ = (_e669_ & 2139095040u);
                    if (_e670_ == 2139095040u) {
                        phi_1601_ = true;
                    } else {
                        phi_1601_ = (_e670_ == 0u);
                    }
                    let _e674_ = phi_1601_;
                    let _e676_ = bitcast<f32>(select(_e669_, 0u, _e674_));
                    let _e677_ = (_e236_ + 5u);
                    if _e237_ {
                        phi_1618_ = 0u;
                    } else {
                        let _e678_ = (_e118_ - 1u);
                        phi_1618_ = select(_e678_, _e677_, (_e677_ < _e678_));
                    }
                    let _e682_ = phi_1618_;
                    if (_e682_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e686_ = seed_params.member[_e682_];
                    let _e687_ = bitcast<u32>(_e686_);
                    let _e688_ = (_e687_ & 2139095040u);
                    if (_e688_ == 2139095040u) {
                        phi_1636_ = true;
                    } else {
                        phi_1636_ = (_e688_ == 0u);
                    }
                    let _e692_ = phi_1636_;
                    let _e694_ = bitcast<f32>(select(_e687_, 0u, _e692_));
                    let _e695_ = (_e236_ + 12u);
                    if _e237_ {
                        phi_1653_ = 0u;
                    } else {
                        let _e696_ = (_e118_ - 1u);
                        phi_1653_ = select(_e696_, _e695_, (_e695_ < _e696_));
                    }
                    let _e700_ = phi_1653_;
                    if (_e700_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e704_ = seed_params.member[_e700_];
                    let _e705_ = bitcast<u32>(_e704_);
                    let _e706_ = (_e705_ & 2139095040u);
                    let _e707_ = (_e706_ == 2139095040u);
                    if _e707_ {
                        if ((_e705_ & 2147483647u) > 2139095040u) {
                            phi_1705_ = f32();
                            phi_1706_ = true;
                        } else {
                            let _e722_ = ((_e705_ & 2147483648u) == 0u);
                            if _e722_ {
                                phi_1703_ = 1f;
                            } else {
                                phi_1703_ = f32();
                            }
                            let _e724_ = phi_1703_;
                            phi_1705_ = _e724_;
                            phi_1706_ = select(true, false, _e722_);
                        }
                        let _e727_ = phi_1705_;
                        let _e729_ = phi_1706_;
                        if _e729_ {
                            phi_1726_ = -1f;
                        } else {
                            phi_1726_ = _e727_;
                        }
                        let _e731_ = phi_1726_;
                        phi_1785_ = _e731_;
                    } else {
                        if (_e704_ < -1f) {
                            phi_1784_ = -1f;
                        } else {
                            if (_e704_ > 1f) {
                                phi_1783_ = 1f;
                            } else {
                                if _e707_ {
                                    phi_1777_ = true;
                                } else {
                                    phi_1777_ = (_e706_ == 0u);
                                }
                                let _e712_ = phi_1777_;
                                phi_1783_ = bitcast<f32>(select(_e705_, 0u, _e712_));
                            }
                            let _e716_ = phi_1783_;
                            phi_1784_ = _e716_;
                        }
                        let _e718_ = phi_1784_;
                        phi_1785_ = _e718_;
                    }
                    let _e733_ = phi_1785_;
                    let _e734_ = (_e236_ + 13u);
                    if _e237_ {
                        phi_1797_ = 0u;
                    } else {
                        let _e735_ = (_e118_ - 1u);
                        phi_1797_ = select(_e735_, _e734_, (_e734_ < _e735_));
                    }
                    let _e739_ = phi_1797_;
                    if (_e739_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e743_ = seed_params.member[_e739_];
                    let _e744_ = bitcast<u32>(_e743_);
                    let _e745_ = (_e744_ & 2139095040u);
                    let _e746_ = (_e745_ == 2139095040u);
                    if _e746_ {
                        if ((_e744_ & 2147483647u) > 2139095040u) {
                            phi_1849_ = f32();
                            phi_1850_ = true;
                        } else {
                            let _e761_ = ((_e744_ & 2147483648u) == 0u);
                            if _e761_ {
                                phi_1847_ = 4f;
                            } else {
                                phi_1847_ = f32();
                            }
                            let _e763_ = phi_1847_;
                            phi_1849_ = _e763_;
                            phi_1850_ = select(true, false, _e761_);
                        }
                        let _e766_ = phi_1849_;
                        let _e768_ = phi_1850_;
                        if _e768_ {
                            phi_1870_ = -4f;
                        } else {
                            phi_1870_ = _e766_;
                        }
                        let _e770_ = phi_1870_;
                        phi_1929_ = _e770_;
                    } else {
                        if (_e743_ < -4f) {
                            phi_1928_ = -4f;
                        } else {
                            if (_e743_ > 4f) {
                                phi_1927_ = 4f;
                            } else {
                                if _e746_ {
                                    phi_1921_ = true;
                                } else {
                                    phi_1921_ = (_e745_ == 0u);
                                }
                                let _e751_ = phi_1921_;
                                phi_1927_ = bitcast<f32>(select(_e744_, 0u, _e751_));
                            }
                            let _e755_ = phi_1927_;
                            phi_1928_ = _e755_;
                        }
                        let _e757_ = phi_1928_;
                        phi_1929_ = _e757_;
                    }
                    let _e772_ = phi_1929_;
                    let _e773_ = (_e236_ + 14u);
                    if _e237_ {
                        phi_1941_ = 0u;
                    } else {
                        let _e774_ = (_e118_ - 1u);
                        phi_1941_ = select(_e774_, _e773_, (_e773_ < _e774_));
                    }
                    let _e778_ = phi_1941_;
                    if (_e778_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e782_ = seed_params.member[_e778_];
                    let _e783_ = bitcast<u32>(_e782_);
                    let _e784_ = (_e783_ & 2139095040u);
                    let _e785_ = (_e784_ == 2139095040u);
                    if _e785_ {
                        if ((_e783_ & 2147483647u) > 2139095040u) {
                            phi_1993_ = f32();
                            phi_1994_ = true;
                        } else {
                            let _e800_ = ((_e783_ & 2147483648u) == 0u);
                            if _e800_ {
                                phi_1991_ = 64f;
                            } else {
                                phi_1991_ = f32();
                            }
                            let _e802_ = phi_1991_;
                            phi_1993_ = _e802_;
                            phi_1994_ = select(true, false, _e800_);
                        }
                        let _e805_ = phi_1993_;
                        let _e807_ = phi_1994_;
                        if _e807_ {
                            phi_2014_ = -64f;
                        } else {
                            phi_2014_ = _e805_;
                        }
                        let _e809_ = phi_2014_;
                        phi_2073_ = _e809_;
                    } else {
                        if (_e782_ < -64f) {
                            phi_2072_ = -64f;
                        } else {
                            if (_e782_ > 64f) {
                                phi_2071_ = 64f;
                            } else {
                                if _e785_ {
                                    phi_2065_ = true;
                                } else {
                                    phi_2065_ = (_e784_ == 0u);
                                }
                                let _e790_ = phi_2065_;
                                phi_2071_ = bitcast<f32>(select(_e783_, 0u, _e790_));
                            }
                            let _e794_ = phi_2071_;
                            phi_2072_ = _e794_;
                        }
                        let _e796_ = phi_2072_;
                        phi_2073_ = _e796_;
                    }
                    let _e811_ = phi_2073_;
                    let _e812_ = (_e236_ + 15u);
                    if _e237_ {
                        phi_2085_ = 0u;
                    } else {
                        let _e813_ = (_e118_ - 1u);
                        phi_2085_ = select(_e813_, _e812_, (_e812_ < _e813_));
                    }
                    let _e817_ = phi_2085_;
                    if (_e817_ < _e118_) {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e821_ = seed_params.member[_e817_];
                    let _e822_ = bitcast<u32>(_e821_);
                    let _e823_ = (_e822_ & 2139095040u);
                    let _e824_ = (_e823_ == 2139095040u);
                    if _e824_ {
                        if ((_e822_ & 2147483647u) > 2139095040u) {
                            phi_2137_ = f32();
                            phi_2138_ = true;
                        } else {
                            let _e839_ = ((_e822_ & 2147483648u) == 0u);
                            if _e839_ {
                                phi_2135_ = 64f;
                            } else {
                                phi_2135_ = f32();
                            }
                            let _e841_ = phi_2135_;
                            phi_2137_ = _e841_;
                            phi_2138_ = select(true, false, _e839_);
                        }
                        let _e844_ = phi_2137_;
                        let _e846_ = phi_2138_;
                        if _e846_ {
                            phi_2158_ = -64f;
                        } else {
                            phi_2158_ = _e844_;
                        }
                        let _e848_ = phi_2158_;
                        phi_2217_ = _e848_;
                    } else {
                        if (_e821_ < -64f) {
                            phi_2216_ = -64f;
                        } else {
                            if (_e821_ > 64f) {
                                phi_2215_ = 64f;
                            } else {
                                if _e824_ {
                                    phi_2209_ = true;
                                } else {
                                    phi_2209_ = (_e823_ == 0u);
                                }
                                let _e829_ = phi_2209_;
                                phi_2215_ = bitcast<f32>(select(_e822_, 0u, _e829_));
                            }
                            let _e833_ = phi_2215_;
                            phi_2216_ = _e833_;
                        }
                        let _e835_ = phi_2216_;
                        phi_2217_ = _e835_;
                    }
                    let _e850_ = phi_2217_;
                    if _e157_ {
                        let _e852_ = (_e592_ + _e592_);
                        let _e853_ = (_e593_ + _e593_);
                        phi_2239_ = type_22_(((_e591_ + _e591_) - 1f), -(_e852_), _e852_, (1f - _e853_), (_e852_ * _e193_.x), (_e853_ * _e193_.x));
                    } else {
                        phi_2239_ = type_22_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e862_ = phi_2239_;
                    let _e864_ = select(255u, _e111_, (_e111_ < 255u));
                    let _e866_ = (_e864_ < 1536u);
                    if _e866_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e864_] = _e862_.member;
                    let _e869_ = (256u + _e864_);
                    let _e870_ = (_e869_ < 1536u);
                    if _e870_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e869_] = _e862_.member_1_;
                    let _e873_ = (512u + _e864_);
                    let _e874_ = (_e873_ < 1536u);
                    if _e874_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e873_] = _e862_.member_2_;
                    let _e877_ = (768u + _e864_);
                    let _e878_ = (_e877_ < 1536u);
                    if _e878_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e877_] = _e862_.member_3_;
                    let _e881_ = (1024u + _e864_);
                    let _e882_ = (_e881_ < 1536u);
                    if _e882_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e881_] = _e862_.member_4_;
                    let _e885_ = (1280u + _e864_);
                    let _e886_ = (_e885_ < 1536u);
                    if _e886_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e885_] = _e862_.member_5_;
                    workgroupBarrier();
                    phi_2299_ = 1u;
                    loop {
                        let _e889_ = phi_2299_;
                        let _e890_ = (_e889_ < 256u);
                        if _e890_ {
                            let _e892_ = (_e889_ << bitcast<u32>(1i));
                            let _e895_ = (((_e111_ + 1u) * _e892_) - 1u);
                            if (_e895_ < 256u) {
                                let _e897_ = (_e895_ - _e889_);
                                let _e899_ = select(255u, _e897_, (_e897_ < 255u));
                                if (_e899_ < 1536u) {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e902_ = scan_scratch[_e899_];
                                let _e903_ = (256u + _e899_);
                                if (_e903_ < 1536u) {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e906_ = scan_scratch[_e903_];
                                let _e907_ = (512u + _e899_);
                                if (_e907_ < 1536u) {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e910_ = scan_scratch[_e907_];
                                let _e911_ = (768u + _e899_);
                                if (_e911_ < 1536u) {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e914_ = scan_scratch[_e911_];
                                let _e915_ = (1024u + _e899_);
                                if (_e915_ < 1536u) {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e918_ = scan_scratch[_e915_];
                                let _e919_ = (1280u + _e899_);
                                if (_e919_ < 1536u) {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e922_ = scan_scratch[_e919_];
                                let _e924_ = select(255u, _e895_, (_e895_ < 255u));
                                let _e925_ = (_e924_ < 1536u);
                                if _e925_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e927_ = scan_scratch[_e924_];
                                let _e928_ = (256u + _e924_);
                                let _e929_ = (_e928_ < 1536u);
                                if _e929_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e931_ = scan_scratch[_e928_];
                                let _e932_ = (512u + _e924_);
                                let _e933_ = (_e932_ < 1536u);
                                if _e933_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e935_ = scan_scratch[_e932_];
                                let _e936_ = (768u + _e924_);
                                let _e937_ = (_e936_ < 1536u);
                                if _e937_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e939_ = scan_scratch[_e936_];
                                let _e940_ = (1024u + _e924_);
                                let _e941_ = (_e940_ < 1536u);
                                if _e941_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e943_ = scan_scratch[_e940_];
                                let _e944_ = (1280u + _e924_);
                                let _e945_ = (_e944_ < 1536u);
                                if _e945_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                let _e947_ = scan_scratch[_e944_];
                                if _e925_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                scan_scratch[_e924_] = ((_e927_ * _e902_) + (_e931_ * _e910_));
                                if _e929_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                scan_scratch[_e928_] = ((_e927_ * _e906_) + (_e931_ * _e914_));
                                if _e933_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                scan_scratch[_e932_] = ((_e935_ * _e902_) + (_e939_ * _e910_));
                                if _e937_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                scan_scratch[_e936_] = ((_e935_ * _e906_) + (_e939_ * _e914_));
                                if _e941_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                scan_scratch[_e940_] = (((_e927_ * _e918_) + (_e931_ * _e922_)) + _e943_);
                                if _e945_ {
                                } else {
                                    phi_10197_ = true;
                                    break;
                                }
                                scan_scratch[_e944_] = (((_e935_ * _e918_) + (_e939_ * _e922_)) + _e947_);
                            }
                            workgroupBarrier();
                            phi_2300_ = _e892_;
                        } else {
                            phi_2300_ = u32();
                        }
                        let _e969_ = phi_2300_;
                        phi_2299_ = _e969_;
                        phi_10197_ = _e196_;
                        if !(_e890_) {
                            break;
                        }
                    }
                    let _e972_ = phi_10197_;
                    phi_10753_ = _e972_;
                    if _e972_ {
                        break;
                    }
                    let _e973_ = (_e111_ == 255u);
                    if _e973_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_2562_ = 128u;
                    loop {
                        let _e981_ = phi_2562_;
                        let _e982_ = (_e981_ > 0u);
                        if _e982_ {
                            let _e987_ = (((_e111_ + 1u) * (_e981_ << bitcast<u32>(1i))) - 1u);
                            if (_e987_ < 256u) {
                                let _e989_ = (_e987_ - _e981_);
                                let _e991_ = select(255u, _e989_, (_e989_ < 255u));
                                let _e992_ = (_e991_ < 1536u);
                                if _e992_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e994_ = scan_scratch[_e991_];
                                let _e995_ = (256u + _e991_);
                                let _e996_ = (_e995_ < 1536u);
                                if _e996_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e998_ = scan_scratch[_e995_];
                                let _e999_ = (512u + _e991_);
                                let _e1000_ = (_e999_ < 1536u);
                                if _e1000_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1002_ = scan_scratch[_e999_];
                                let _e1003_ = (768u + _e991_);
                                let _e1004_ = (_e1003_ < 1536u);
                                if _e1004_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1006_ = scan_scratch[_e1003_];
                                let _e1007_ = (1024u + _e991_);
                                let _e1008_ = (_e1007_ < 1536u);
                                if _e1008_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1010_ = scan_scratch[_e1007_];
                                let _e1011_ = (1280u + _e991_);
                                let _e1012_ = (_e1011_ < 1536u);
                                if _e1012_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1014_ = scan_scratch[_e1011_];
                                let _e1016_ = select(255u, _e987_, (_e987_ < 255u));
                                let _e1017_ = (_e1016_ < 1536u);
                                if _e1017_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1019_ = scan_scratch[_e1016_];
                                let _e1020_ = (256u + _e1016_);
                                let _e1021_ = (_e1020_ < 1536u);
                                if _e1021_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1023_ = scan_scratch[_e1020_];
                                let _e1024_ = (512u + _e1016_);
                                let _e1025_ = (_e1024_ < 1536u);
                                if _e1025_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1027_ = scan_scratch[_e1024_];
                                let _e1028_ = (768u + _e1016_);
                                let _e1029_ = (_e1028_ < 1536u);
                                if _e1029_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1031_ = scan_scratch[_e1028_];
                                let _e1032_ = (1024u + _e1016_);
                                let _e1033_ = (_e1032_ < 1536u);
                                if _e1033_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1035_ = scan_scratch[_e1032_];
                                let _e1036_ = (1280u + _e1016_);
                                let _e1037_ = (_e1036_ < 1536u);
                                if _e1037_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                let _e1039_ = scan_scratch[_e1036_];
                                if _e992_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e991_] = _e1019_;
                                if _e996_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e995_] = _e1023_;
                                if _e1000_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e999_] = _e1027_;
                                if _e1004_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1003_] = _e1031_;
                                if _e1008_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1007_] = _e1035_;
                                if _e1012_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1011_] = _e1039_;
                                if _e1017_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1016_] = ((_e994_ * _e1019_) + (_e998_ * _e1027_));
                                if _e1021_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1020_] = ((_e994_ * _e1023_) + (_e998_ * _e1031_));
                                if _e1025_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1024_] = ((_e1002_ * _e1019_) + (_e1006_ * _e1027_));
                                if _e1029_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1028_] = ((_e1002_ * _e1023_) + (_e1006_ * _e1031_));
                                if _e1033_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1032_] = (((_e994_ * _e1035_) + (_e998_ * _e1039_)) + _e1010_);
                                if _e1037_ {
                                } else {
                                    phi_10498_ = true;
                                    break;
                                }
                                scan_scratch[_e1036_] = (((_e1002_ * _e1035_) + (_e1006_ * _e1039_)) + _e1014_);
                            }
                            workgroupBarrier();
                            phi_2563_ = (_e981_ >> bitcast<u32>(1i));
                        } else {
                            phi_2563_ = u32();
                        }
                        let _e1063_ = phi_2563_;
                        phi_2562_ = _e1063_;
                        phi_10498_ = _e972_;
                        if !(_e982_) {
                            break;
                        }
                    }
                    let _e1066_ = phi_10498_;
                    phi_10753_ = _e1066_;
                    if _e1066_ {
                        break;
                    }
                    if _e866_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e870_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e874_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e878_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e882_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e1067_ = scan_scratch[_e881_];
                    if _e886_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e1068_ = scan_scratch[_e885_];
                    workgroupBarrier();
                    if _e157_ {
                        let _e1069_ = (_e193_.x - _e1068_);
                        let _e1072_ = ((_e591_ * _e1067_) + (_e592_ * _e1069_));
                        let _e1076_ = ((_e1068_ + (_e592_ * _e1067_)) + (_e593_ * _e1069_));
                        let _e1087_ = bitcast<u32>(((((_e640_ * _e1076_) + (_e658_ * _e1072_)) + (_e676_ * ((_e193_.x - (_e577_ * _e1072_)) - _e1076_))) + (_e694_ * _e193_.x)));
                        let _e1088_ = (_e1087_ & 2139095040u);
                        if (_e1088_ == 2139095040u) {
                            phi_3105_ = true;
                        } else {
                            phi_3105_ = (_e1088_ == 0u);
                        }
                        let _e1092_ = phi_3105_;
                        let _e1095_ = bitcast<u32>(_e1072_);
                        let _e1096_ = (_e1095_ & 2139095040u);
                        if (_e1096_ == 2139095040u) {
                            phi_3121_ = true;
                        } else {
                            phi_3121_ = (_e1096_ == 0u);
                        }
                        let _e1100_ = phi_3121_;
                        phi_3129_ = type_21_(bitcast<f32>(select(_e1087_, 0u, _e1092_)), bitcast<f32>(select(_e1095_, 0u, _e1100_)));
                    } else {
                        phi_3129_ = type_21_(0f, 0f);
                    }
                    let _e1105_ = phi_3129_;
                    if _e157_ {
                        let _e1109_ = (_e592_ + _e592_);
                        let _e1110_ = (_e593_ + _e593_);
                        phi_3152_ = type_22_(((_e591_ + _e591_) - 1f), -(_e1109_), _e1109_, (1f - _e1110_), (_e1109_ * _e193_.y), (_e1110_ * _e193_.y));
                    } else {
                        phi_3152_ = type_22_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e1119_ = phi_3152_;
                    if _e866_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e864_] = _e1119_.member;
                    if _e870_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e869_] = _e1119_.member_1_;
                    if _e874_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e873_] = _e1119_.member_2_;
                    if _e878_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e877_] = _e1119_.member_3_;
                    if _e882_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e881_] = _e1119_.member_4_;
                    if _e886_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e885_] = _e1119_.member_5_;
                    workgroupBarrier();
                    phi_3212_ = 1u;
                    loop {
                        let _e1127_ = phi_3212_;
                        let _e1128_ = (_e1127_ < 256u);
                        if _e1128_ {
                            let _e1130_ = (_e1127_ << bitcast<u32>(1i));
                            let _e1133_ = (((_e111_ + 1u) * _e1130_) - 1u);
                            if (_e1133_ < 256u) {
                                let _e1135_ = (_e1133_ - _e1127_);
                                let _e1137_ = select(255u, _e1135_, (_e1135_ < 255u));
                                if (_e1137_ < 1536u) {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1140_ = scan_scratch[_e1137_];
                                let _e1141_ = (256u + _e1137_);
                                if (_e1141_ < 1536u) {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1144_ = scan_scratch[_e1141_];
                                let _e1145_ = (512u + _e1137_);
                                if (_e1145_ < 1536u) {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1148_ = scan_scratch[_e1145_];
                                let _e1149_ = (768u + _e1137_);
                                if (_e1149_ < 1536u) {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1152_ = scan_scratch[_e1149_];
                                let _e1153_ = (1024u + _e1137_);
                                if (_e1153_ < 1536u) {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1156_ = scan_scratch[_e1153_];
                                let _e1157_ = (1280u + _e1137_);
                                if (_e1157_ < 1536u) {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1160_ = scan_scratch[_e1157_];
                                let _e1162_ = select(255u, _e1133_, (_e1133_ < 255u));
                                let _e1163_ = (_e1162_ < 1536u);
                                if _e1163_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1165_ = scan_scratch[_e1162_];
                                let _e1166_ = (256u + _e1162_);
                                let _e1167_ = (_e1166_ < 1536u);
                                if _e1167_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1169_ = scan_scratch[_e1166_];
                                let _e1170_ = (512u + _e1162_);
                                let _e1171_ = (_e1170_ < 1536u);
                                if _e1171_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1173_ = scan_scratch[_e1170_];
                                let _e1174_ = (768u + _e1162_);
                                let _e1175_ = (_e1174_ < 1536u);
                                if _e1175_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1177_ = scan_scratch[_e1174_];
                                let _e1178_ = (1024u + _e1162_);
                                let _e1179_ = (_e1178_ < 1536u);
                                if _e1179_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1181_ = scan_scratch[_e1178_];
                                let _e1182_ = (1280u + _e1162_);
                                let _e1183_ = (_e1182_ < 1536u);
                                if _e1183_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                let _e1185_ = scan_scratch[_e1182_];
                                if _e1163_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                scan_scratch[_e1162_] = ((_e1165_ * _e1140_) + (_e1169_ * _e1148_));
                                if _e1167_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                scan_scratch[_e1166_] = ((_e1165_ * _e1144_) + (_e1169_ * _e1152_));
                                if _e1171_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                scan_scratch[_e1170_] = ((_e1173_ * _e1140_) + (_e1177_ * _e1148_));
                                if _e1175_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                scan_scratch[_e1174_] = ((_e1173_ * _e1144_) + (_e1177_ * _e1152_));
                                if _e1179_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                scan_scratch[_e1178_] = (((_e1165_ * _e1156_) + (_e1169_ * _e1160_)) + _e1181_);
                                if _e1183_ {
                                } else {
                                    phi_10513_ = true;
                                    break;
                                }
                                scan_scratch[_e1182_] = (((_e1173_ * _e1156_) + (_e1177_ * _e1160_)) + _e1185_);
                            }
                            workgroupBarrier();
                            phi_3213_ = _e1130_;
                        } else {
                            phi_3213_ = u32();
                        }
                        let _e1207_ = phi_3213_;
                        phi_3212_ = _e1207_;
                        phi_10513_ = _e1066_;
                        if !(_e1128_) {
                            break;
                        }
                    }
                    let _e1210_ = phi_10513_;
                    phi_10753_ = _e1210_;
                    if _e1210_ {
                        break;
                    }
                    if _e973_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_3475_ = 128u;
                    loop {
                        let _e1218_ = phi_3475_;
                        let _e1219_ = (_e1218_ > 0u);
                        if _e1219_ {
                            let _e1224_ = (((_e111_ + 1u) * (_e1218_ << bitcast<u32>(1i))) - 1u);
                            if (_e1224_ < 256u) {
                                let _e1226_ = (_e1224_ - _e1218_);
                                let _e1228_ = select(255u, _e1226_, (_e1226_ < 255u));
                                let _e1229_ = (_e1228_ < 1536u);
                                if _e1229_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1231_ = scan_scratch[_e1228_];
                                let _e1232_ = (256u + _e1228_);
                                let _e1233_ = (_e1232_ < 1536u);
                                if _e1233_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1235_ = scan_scratch[_e1232_];
                                let _e1236_ = (512u + _e1228_);
                                let _e1237_ = (_e1236_ < 1536u);
                                if _e1237_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1239_ = scan_scratch[_e1236_];
                                let _e1240_ = (768u + _e1228_);
                                let _e1241_ = (_e1240_ < 1536u);
                                if _e1241_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1243_ = scan_scratch[_e1240_];
                                let _e1244_ = (1024u + _e1228_);
                                let _e1245_ = (_e1244_ < 1536u);
                                if _e1245_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1247_ = scan_scratch[_e1244_];
                                let _e1248_ = (1280u + _e1228_);
                                let _e1249_ = (_e1248_ < 1536u);
                                if _e1249_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1251_ = scan_scratch[_e1248_];
                                let _e1253_ = select(255u, _e1224_, (_e1224_ < 255u));
                                let _e1254_ = (_e1253_ < 1536u);
                                if _e1254_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1256_ = scan_scratch[_e1253_];
                                let _e1257_ = (256u + _e1253_);
                                let _e1258_ = (_e1257_ < 1536u);
                                if _e1258_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1260_ = scan_scratch[_e1257_];
                                let _e1261_ = (512u + _e1253_);
                                let _e1262_ = (_e1261_ < 1536u);
                                if _e1262_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1264_ = scan_scratch[_e1261_];
                                let _e1265_ = (768u + _e1253_);
                                let _e1266_ = (_e1265_ < 1536u);
                                if _e1266_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1268_ = scan_scratch[_e1265_];
                                let _e1269_ = (1024u + _e1253_);
                                let _e1270_ = (_e1269_ < 1536u);
                                if _e1270_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1272_ = scan_scratch[_e1269_];
                                let _e1273_ = (1280u + _e1253_);
                                let _e1274_ = (_e1273_ < 1536u);
                                if _e1274_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                let _e1276_ = scan_scratch[_e1273_];
                                if _e1229_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1228_] = _e1256_;
                                if _e1233_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1232_] = _e1260_;
                                if _e1237_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1236_] = _e1264_;
                                if _e1241_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1240_] = _e1268_;
                                if _e1245_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1244_] = _e1272_;
                                if _e1249_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1248_] = _e1276_;
                                if _e1254_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1253_] = ((_e1231_ * _e1256_) + (_e1235_ * _e1264_));
                                if _e1258_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1257_] = ((_e1231_ * _e1260_) + (_e1235_ * _e1268_));
                                if _e1262_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1261_] = ((_e1239_ * _e1256_) + (_e1243_ * _e1264_));
                                if _e1266_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1265_] = ((_e1239_ * _e1260_) + (_e1243_ * _e1268_));
                                if _e1270_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1269_] = (((_e1231_ * _e1272_) + (_e1235_ * _e1276_)) + _e1247_);
                                if _e1274_ {
                                } else {
                                    phi_10533_ = true;
                                    break;
                                }
                                scan_scratch[_e1273_] = (((_e1239_ * _e1272_) + (_e1243_ * _e1276_)) + _e1251_);
                            }
                            workgroupBarrier();
                            phi_3476_ = (_e1218_ >> bitcast<u32>(1i));
                        } else {
                            phi_3476_ = u32();
                        }
                        let _e1300_ = phi_3476_;
                        phi_3475_ = _e1300_;
                        phi_10533_ = _e1210_;
                        if !(_e1219_) {
                            break;
                        }
                    }
                    let _e1303_ = phi_10533_;
                    phi_10753_ = _e1303_;
                    if _e1303_ {
                        break;
                    }
                    if _e866_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e870_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e874_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e878_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    if _e882_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e1304_ = scan_scratch[_e881_];
                    if _e886_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    let _e1305_ = scan_scratch[_e885_];
                    workgroupBarrier();
                    if _e157_ {
                        let _e1306_ = (_e193_.y - _e1305_);
                        let _e1309_ = ((_e591_ * _e1304_) + (_e592_ * _e1306_));
                        let _e1313_ = ((_e1305_ + (_e592_ * _e1304_)) + (_e593_ * _e1306_));
                        let _e1324_ = bitcast<u32>(((((_e640_ * _e1313_) + (_e658_ * _e1309_)) + (_e676_ * ((_e193_.y - (_e577_ * _e1309_)) - _e1313_))) + (_e694_ * _e193_.y)));
                        let _e1325_ = (_e1324_ & 2139095040u);
                        if (_e1325_ == 2139095040u) {
                            phi_4018_ = true;
                        } else {
                            phi_4018_ = (_e1325_ == 0u);
                        }
                        let _e1329_ = phi_4018_;
                        let _e1332_ = bitcast<u32>(_e1309_);
                        let _e1333_ = (_e1332_ & 2139095040u);
                        if (_e1333_ == 2139095040u) {
                            phi_4034_ = true;
                        } else {
                            phi_4034_ = (_e1333_ == 0u);
                        }
                        let _e1337_ = phi_4034_;
                        phi_4042_ = type_21_(bitcast<f32>(select(_e1324_, 0u, _e1329_)), bitcast<f32>(select(_e1332_, 0u, _e1337_)));
                    } else {
                        phi_4042_ = type_21_(0f, 0f);
                    }
                    let _e1342_ = phi_4042_;
                    if _e866_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e864_] = _e1105_.member;
                    if _e870_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e869_] = _e1342_.member;
                    if _e874_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e873_] = _e1105_.member_1_;
                    if _e878_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e877_] = _e1342_.member_1_;
                    if _e882_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e881_] = 0f;
                    if _e886_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e885_] = 0f;
                    workgroupBarrier();
                    if (_e111_ == 0u) {
                        phi_4107_ = type_15_(0u, select(_e153_, 256u, (256u < _e153_)));
                        phi_4110_ = type_23_(array<f32, 4>(0f, 0f, 0f, 0f));
                        phi_4112_ = type_23_(array<f32, 4>(0f, 0f, 0f, 0f));
                        loop {
                            let _e1348_ = phi_4107_;
                            let _e1350_ = phi_4110_;
                            let _e1352_ = phi_4112_;
                            if (_e1348_.member < _e1348_.member_1_) {
                                phi_4108_ = type_15_((_e1348_.member + 1u), _e1348_.member_1_);
                                phi_4135_ = type_15_(1u, _e1348_.member);
                            } else {
                                phi_4108_ = _e1348_;
                                phi_4135_ = type_15_(0u, u32());
                            }
                            let _e1365_ = phi_4108_;
                            let _e1367_ = phi_4135_;
                            let _e1371_ = (bitcast<i32>(_e1367_.member) != 0i);
                            if _e1371_ {
                                if (_e1367_.member_1_ < 1536u) {
                                } else {
                                    phi_10548_ = true;
                                    break;
                                }
                                let _e1374_ = scan_scratch[_e1367_.member_1_];
                                let _e1375_ = (256u + _e1367_.member_1_);
                                if (_e1375_ < 1536u) {
                                } else {
                                    phi_10548_ = true;
                                    break;
                                }
                                let _e1378_ = scan_scratch[_e1375_];
                                let _e1379_ = (512u + _e1367_.member_1_);
                                if (_e1379_ < 1536u) {
                                } else {
                                    phi_10548_ = true;
                                    break;
                                }
                                let _e1382_ = scan_scratch[_e1379_];
                                let _e1383_ = (768u + _e1367_.member_1_);
                                if (_e1383_ < 1536u) {
                                } else {
                                    phi_10548_ = true;
                                    break;
                                }
                                let _e1386_ = scan_scratch[_e1383_];
                                let _e1387_ = (_e733_ * _e1374_);
                                let _e1388_ = (_e733_ * _e1378_);
                                let _e1391_ = bitcast<u32>((_e622_ * _e1382_));
                                let _e1392_ = (_e1391_ & 2139095040u);
                                let _e1393_ = (_e1392_ == 2139095040u);
                                if _e1393_ {
                                    phi_4188_ = true;
                                } else {
                                    phi_4188_ = (_e1392_ == 0u);
                                }
                                let _e1396_ = phi_4188_;
                                if _e1393_ {
                                    phi_4204_ = true;
                                } else {
                                    phi_4204_ = (_e1392_ == 0u);
                                }
                                let _e1401_ = phi_4204_;
                                if _e1393_ {
                                    phi_4220_ = true;
                                } else {
                                    phi_4220_ = (_e1392_ == 0u);
                                }
                                let _e1406_ = phi_4220_;
                                if _e1393_ {
                                    phi_4236_ = true;
                                } else {
                                    phi_4236_ = (_e1392_ == 0u);
                                }
                                let _e1411_ = phi_4236_;
                                let _e1416_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1391_, 0u, _e1396_))) & 2147483647u));
                                let _e1419_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1391_, 0u, _e1401_))) & 2147483647u));
                                let _e1422_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1391_, 0u, _e1406_))) & 2147483647u));
                                let _e1425_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1391_, 0u, _e1411_))) & 2147483647u));
                                if (_e1416_ != _e1416_) {
                                    phi_7537_ = true;
                                } else {
                                    phi_7537_ = (1000000000000000000f <= _e1416_);
                                }
                                let _e1429_ = phi_7537_;
                                if (_e1419_ != _e1419_) {
                                    phi_7552_ = true;
                                } else {
                                    phi_7552_ = (1000000000000000000f <= _e1419_);
                                }
                                let _e1434_ = phi_7552_;
                                if (_e1422_ != _e1422_) {
                                    phi_7567_ = true;
                                } else {
                                    phi_7567_ = (1000000000000000000f <= _e1422_);
                                }
                                let _e1439_ = phi_7567_;
                                if (_e1425_ != _e1425_) {
                                    phi_7582_ = true;
                                } else {
                                    phi_7582_ = (1000000000000000000f <= _e1425_);
                                }
                                let _e1444_ = phi_7582_;
                                let _e1450_ = (_e555_ <= 0f);
                                let _e1451_ = select((_e555_ * select(_e1416_, 1000000000000000000f, _e1429_)), 0f, _e1450_);
                                let _e1452_ = select((_e555_ * select(_e1419_, 1000000000000000000f, _e1434_)), 0f, _e1450_);
                                let _e1453_ = select((_e555_ * select(_e1422_, 1000000000000000000f, _e1439_)), 0f, _e1450_);
                                let _e1454_ = select((_e555_ * select(_e1425_, 1000000000000000000f, _e1444_)), 0f, _e1450_);
                                let _e1455_ = (1f + _e1451_);
                                let _e1456_ = (1f + _e1452_);
                                let _e1457_ = (1f + _e1453_);
                                let _e1458_ = (1f + _e1454_);
                                let _e1461_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1455_)));
                                let _e1464_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1456_)));
                                let _e1467_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1457_)));
                                let _e1470_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1458_)));
                                let _e1480_ = (_e1461_ + (_e1461_ * (1f - (_e1455_ * _e1461_))));
                                let _e1482_ = (_e1464_ + (_e1464_ * (1f - (_e1456_ * _e1464_))));
                                let _e1484_ = (_e1467_ + (_e1467_ * (1f - (_e1457_ * _e1467_))));
                                let _e1486_ = (_e1470_ + (_e1470_ * (1f - (_e1458_ * _e1470_))));
                                let _e1512_ = (1f - (_e543_ * select((_e1480_ + (_e1480_ * (1f - (_e1455_ * _e1480_)))), 1f, (_e1451_ <= 0f))));
                                let _e1514_ = (1f - (_e543_ * select((_e1482_ + (_e1482_ * (1f - (_e1456_ * _e1482_)))), 1f, (_e1452_ <= 0f))));
                                let _e1516_ = (1f - (_e543_ * select((_e1484_ + (_e1484_ * (1f - (_e1457_ * _e1484_)))), 1f, (_e1453_ <= 0f))));
                                let _e1518_ = (1f - (_e543_ * select((_e1486_ + (_e1486_ * (1f - (_e1458_ * _e1486_)))), 1f, (_e1454_ <= 0f))));
                                if (_e1512_ != _e1512_) {
                                    phi_7597_ = true;
                                } else {
                                    phi_7597_ = (0f >= _e1512_);
                                }
                                let _e1522_ = phi_7597_;
                                let _e1523_ = select(_e1512_, 0f, _e1522_);
                                if (_e1514_ != _e1514_) {
                                    phi_7612_ = true;
                                } else {
                                    phi_7612_ = (0f >= _e1514_);
                                }
                                let _e1527_ = phi_7612_;
                                let _e1528_ = select(_e1514_, 0f, _e1527_);
                                if (_e1516_ != _e1516_) {
                                    phi_7627_ = true;
                                } else {
                                    phi_7627_ = (0f >= _e1516_);
                                }
                                let _e1532_ = phi_7627_;
                                let _e1533_ = select(_e1516_, 0f, _e1532_);
                                if (_e1518_ != _e1518_) {
                                    phi_7642_ = true;
                                } else {
                                    phi_7642_ = (0f >= _e1518_);
                                }
                                let _e1537_ = phi_7642_;
                                let _e1538_ = select(_e1518_, 0f, _e1537_);
                                if (_e1523_ != _e1523_) {
                                    phi_7657_ = true;
                                } else {
                                    phi_7657_ = (1f <= _e1523_);
                                }
                                let _e1542_ = phi_7657_;
                                let _e1543_ = select(_e1523_, 1f, _e1542_);
                                if (_e1528_ != _e1528_) {
                                    phi_7672_ = true;
                                } else {
                                    phi_7672_ = (1f <= _e1528_);
                                }
                                let _e1547_ = phi_7672_;
                                let _e1548_ = select(_e1528_, 1f, _e1547_);
                                if (_e1533_ != _e1533_) {
                                    phi_7687_ = true;
                                } else {
                                    phi_7687_ = (1f <= _e1533_);
                                }
                                let _e1552_ = phi_7687_;
                                let _e1553_ = select(_e1533_, 1f, _e1552_);
                                if (_e1538_ != _e1538_) {
                                    phi_7702_ = true;
                                } else {
                                    phi_7702_ = (1f <= _e1538_);
                                }
                                let _e1557_ = phi_7702_;
                                let _e1558_ = select(_e1538_, 1f, _e1557_);
                                let _e1561_ = (_e537_ * _e1352_.member[0]);
                                let _e1564_ = (_e537_ * _e1352_.member[1]);
                                let _e1567_ = (_e537_ * _e1352_.member[2]);
                                let _e1570_ = (_e537_ * _e1352_.member[3]);
                                let _e1571_ = (1f - _e561_);
                                let _e1572_ = (select(select(select(0f, 1f, (1f <= _e477_)), _e477_, (_e477_ <= 1f)), 0f, (_e477_ <= 0f)) * _e1571_);
                                let _e1574_ = (_e1387_ - (_e1572_ * _e1561_));
                                let _e1576_ = (_e1387_ - (_e1572_ * _e1564_));
                                let _e1578_ = (_e1387_ - (_e1572_ * _e1567_));
                                let _e1580_ = (_e1387_ - (_e1572_ * _e1570_));
                                if (_e1574_ != _e1574_) {
                                    phi_7717_ = true;
                                } else {
                                    phi_7717_ = (-1f >= _e1574_);
                                }
                                let _e1584_ = phi_7717_;
                                let _e1585_ = select(_e1574_, -1f, _e1584_);
                                if (_e1576_ != _e1576_) {
                                    phi_7732_ = true;
                                } else {
                                    phi_7732_ = (-1f >= _e1576_);
                                }
                                let _e1589_ = phi_7732_;
                                let _e1590_ = select(_e1576_, -1f, _e1589_);
                                if (_e1578_ != _e1578_) {
                                    phi_7747_ = true;
                                } else {
                                    phi_7747_ = (-1f >= _e1578_);
                                }
                                let _e1594_ = phi_7747_;
                                let _e1595_ = select(_e1578_, -1f, _e1594_);
                                if (_e1580_ != _e1580_) {
                                    phi_7762_ = true;
                                } else {
                                    phi_7762_ = (-1f >= _e1580_);
                                }
                                let _e1599_ = phi_7762_;
                                let _e1600_ = select(_e1580_, -1f, _e1599_);
                                if (_e1585_ != _e1585_) {
                                    phi_7777_ = true;
                                } else {
                                    phi_7777_ = (1f <= _e1585_);
                                }
                                let _e1604_ = phi_7777_;
                                let _e1605_ = select(_e1585_, 1f, _e1604_);
                                if (_e1590_ != _e1590_) {
                                    phi_7792_ = true;
                                } else {
                                    phi_7792_ = (1f <= _e1590_);
                                }
                                let _e1609_ = phi_7792_;
                                let _e1610_ = select(_e1590_, 1f, _e1609_);
                                if (_e1595_ != _e1595_) {
                                    phi_7807_ = true;
                                } else {
                                    phi_7807_ = (1f <= _e1595_);
                                }
                                let _e1614_ = phi_7807_;
                                let _e1615_ = select(_e1595_, 1f, _e1614_);
                                if (_e1600_ != _e1600_) {
                                    phi_7822_ = true;
                                } else {
                                    phi_7822_ = (1f <= _e1600_);
                                }
                                let _e1619_ = phi_7822_;
                                let _e1620_ = select(_e1600_, 1f, _e1619_);
                                let _e1650_ = select(0f, _e1574_, (bitcast<f32>((bitcast<u32>(_e1574_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1652_ = select(0f, _e1576_, (bitcast<f32>((bitcast<u32>(_e1576_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1654_ = select(0f, _e1578_, (bitcast<f32>((bitcast<u32>(_e1578_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1656_ = select(0f, _e1580_, (bitcast<f32>((bitcast<u32>(_e1580_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e1657_ = (_e1650_ * 499.99997f);
                                let _e1658_ = (_e1652_ * 499.99997f);
                                let _e1659_ = (_e1654_ * 499.99997f);
                                let _e1660_ = (_e1656_ * 499.99997f);
                                if (_e1657_ != _e1657_) {
                                    phi_7837_ = true;
                                } else {
                                    phi_7837_ = (0f >= _e1657_);
                                }
                                let _e1664_ = phi_7837_;
                                let _e1665_ = select(_e1657_, 0f, _e1664_);
                                if (_e1658_ != _e1658_) {
                                    phi_7852_ = true;
                                } else {
                                    phi_7852_ = (0f >= _e1658_);
                                }
                                let _e1669_ = phi_7852_;
                                let _e1670_ = select(_e1658_, 0f, _e1669_);
                                if (_e1659_ != _e1659_) {
                                    phi_7867_ = true;
                                } else {
                                    phi_7867_ = (0f >= _e1659_);
                                }
                                let _e1674_ = phi_7867_;
                                let _e1675_ = select(_e1659_, 0f, _e1674_);
                                if (_e1660_ != _e1660_) {
                                    phi_7882_ = true;
                                } else {
                                    phi_7882_ = (0f >= _e1660_);
                                }
                                let _e1679_ = phi_7882_;
                                let _e1680_ = select(_e1660_, 0f, _e1679_);
                                if (_e1665_ != _e1665_) {
                                    phi_7897_ = true;
                                } else {
                                    phi_7897_ = (1f <= _e1665_);
                                }
                                let _e1684_ = phi_7897_;
                                let _e1685_ = select(_e1665_, 1f, _e1684_);
                                if (_e1670_ != _e1670_) {
                                    phi_7912_ = true;
                                } else {
                                    phi_7912_ = (1f <= _e1670_);
                                }
                                let _e1689_ = phi_7912_;
                                let _e1690_ = select(_e1670_, 1f, _e1689_);
                                if (_e1675_ != _e1675_) {
                                    phi_7927_ = true;
                                } else {
                                    phi_7927_ = (1f <= _e1675_);
                                }
                                let _e1694_ = phi_7927_;
                                let _e1695_ = select(_e1675_, 1f, _e1694_);
                                if (_e1680_ != _e1680_) {
                                    phi_7942_ = true;
                                } else {
                                    phi_7942_ = (1f <= _e1680_);
                                }
                                let _e1699_ = phi_7942_;
                                let _e1700_ = select(_e1680_, 1f, _e1699_);
                                let _e1721_ = (_e1650_ - 0.002f);
                                let _e1722_ = (_e1652_ - 0.002f);
                                let _e1723_ = (_e1654_ - 0.002f);
                                let _e1724_ = (_e1656_ - 0.002f);
                                if (_e1721_ != _e1721_) {
                                    phi_7957_ = true;
                                } else {
                                    phi_7957_ = (0f >= _e1721_);
                                }
                                let _e1728_ = phi_7957_;
                                if (_e1722_ != _e1722_) {
                                    phi_7972_ = true;
                                } else {
                                    phi_7972_ = (0f >= _e1722_);
                                }
                                let _e1733_ = phi_7972_;
                                if (_e1723_ != _e1723_) {
                                    phi_7987_ = true;
                                } else {
                                    phi_7987_ = (0f >= _e1723_);
                                }
                                let _e1738_ = phi_7987_;
                                if (_e1724_ != _e1724_) {
                                    phi_8002_ = true;
                                } else {
                                    phi_8002_ = (0f >= _e1724_);
                                }
                                let _e1743_ = phi_8002_;
                                let _e1745_ = ((0.001f * (((_e1685_ * _e1685_) * _e1685_) * (2f - _e1685_))) + select(_e1721_, 0f, _e1728_));
                                let _e1746_ = ((0.001f * (((_e1690_ * _e1690_) * _e1690_) * (2f - _e1690_))) + select(_e1722_, 0f, _e1733_));
                                let _e1747_ = ((0.001f * (((_e1695_ * _e1695_) * _e1695_) * (2f - _e1695_))) + select(_e1723_, 0f, _e1738_));
                                let _e1748_ = ((0.001f * (((_e1700_ * _e1700_) * _e1700_) * (2f - _e1700_))) + select(_e1724_, 0f, _e1743_));
                                let _e1761_ = (_e561_ <= 0f);
                                let _e1786_ = bitcast<u32>((((1f - (_e561_ * _e1543_)) * _e1561_) + (_e1543_ * select(((_e1571_ * _e1745_) + (_e561_ * (_e1605_ * (1f - (0.33333334f * (_e1605_ * _e1605_)))))), _e1745_, _e1761_))));
                                let _e1787_ = (_e1786_ & 2139095040u);
                                if (_e1787_ == 2139095040u) {
                                    phi_4643_ = true;
                                } else {
                                    phi_4643_ = (_e1787_ == 0u);
                                }
                                let _e1791_ = phi_4643_;
                                let _e1793_ = bitcast<f32>(select(_e1786_, 0u, _e1791_));
                                let _e1794_ = bitcast<u32>((((1f - (_e561_ * _e1548_)) * _e1564_) + (_e1548_ * select(((_e1571_ * _e1746_) + (_e561_ * (_e1610_ * (1f - (0.33333334f * (_e1610_ * _e1610_)))))), _e1746_, _e1761_))));
                                let _e1795_ = (_e1794_ & 2139095040u);
                                if (_e1795_ == 2139095040u) {
                                    phi_4659_ = true;
                                } else {
                                    phi_4659_ = (_e1795_ == 0u);
                                }
                                let _e1799_ = phi_4659_;
                                let _e1802_ = bitcast<u32>((((1f - (_e561_ * _e1553_)) * _e1567_) + (_e1553_ * select(((_e1571_ * _e1747_) + (_e561_ * (_e1615_ * (1f - (0.33333334f * (_e1615_ * _e1615_)))))), _e1747_, _e1761_))));
                                let _e1803_ = (_e1802_ & 2139095040u);
                                if (_e1803_ == 2139095040u) {
                                    phi_4675_ = true;
                                } else {
                                    phi_4675_ = (_e1803_ == 0u);
                                }
                                let _e1807_ = phi_4675_;
                                let _e1810_ = bitcast<u32>((((1f - (_e561_ * _e1558_)) * _e1570_) + (_e1558_ * select(((_e1571_ * _e1748_) + (_e561_ * (_e1620_ * (1f - (0.33333334f * (_e1620_ * _e1620_)))))), _e1748_, _e1761_))));
                                let _e1811_ = (_e1810_ & 2139095040u);
                                if (_e1811_ == 2139095040u) {
                                    phi_4691_ = true;
                                } else {
                                    phi_4691_ = (_e1811_ == 0u);
                                }
                                let _e1815_ = phi_4691_;
                                let _e1820_ = bitcast<u32>((_e622_ * _e1386_));
                                let _e1821_ = (_e1820_ & 2139095040u);
                                let _e1822_ = (_e1821_ == 2139095040u);
                                if _e1822_ {
                                    phi_4713_ = true;
                                } else {
                                    phi_4713_ = (_e1821_ == 0u);
                                }
                                let _e1825_ = phi_4713_;
                                if _e1822_ {
                                    phi_4729_ = true;
                                } else {
                                    phi_4729_ = (_e1821_ == 0u);
                                }
                                let _e1830_ = phi_4729_;
                                if _e1822_ {
                                    phi_4745_ = true;
                                } else {
                                    phi_4745_ = (_e1821_ == 0u);
                                }
                                let _e1835_ = phi_4745_;
                                if _e1822_ {
                                    phi_4761_ = true;
                                } else {
                                    phi_4761_ = (_e1821_ == 0u);
                                }
                                let _e1840_ = phi_4761_;
                                let _e1845_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1820_, 0u, _e1825_))) & 2147483647u));
                                let _e1848_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1820_, 0u, _e1830_))) & 2147483647u));
                                let _e1851_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1820_, 0u, _e1835_))) & 2147483647u));
                                let _e1854_ = bitcast<f32>((bitcast<u32>(bitcast<f32>(select(_e1820_, 0u, _e1840_))) & 2147483647u));
                                if (_e1845_ != _e1845_) {
                                    phi_8017_ = true;
                                } else {
                                    phi_8017_ = (1000000000000000000f <= _e1845_);
                                }
                                let _e1858_ = phi_8017_;
                                if (_e1848_ != _e1848_) {
                                    phi_8032_ = true;
                                } else {
                                    phi_8032_ = (1000000000000000000f <= _e1848_);
                                }
                                let _e1863_ = phi_8032_;
                                if (_e1851_ != _e1851_) {
                                    phi_8047_ = true;
                                } else {
                                    phi_8047_ = (1000000000000000000f <= _e1851_);
                                }
                                let _e1868_ = phi_8047_;
                                if (_e1854_ != _e1854_) {
                                    phi_8062_ = true;
                                } else {
                                    phi_8062_ = (1000000000000000000f <= _e1854_);
                                }
                                let _e1873_ = phi_8062_;
                                let _e1879_ = select((_e555_ * select(_e1845_, 1000000000000000000f, _e1858_)), 0f, _e1450_);
                                let _e1880_ = select((_e555_ * select(_e1848_, 1000000000000000000f, _e1863_)), 0f, _e1450_);
                                let _e1881_ = select((_e555_ * select(_e1851_, 1000000000000000000f, _e1868_)), 0f, _e1450_);
                                let _e1882_ = select((_e555_ * select(_e1854_, 1000000000000000000f, _e1873_)), 0f, _e1450_);
                                let _e1883_ = (1f + _e1879_);
                                let _e1884_ = (1f + _e1880_);
                                let _e1885_ = (1f + _e1881_);
                                let _e1886_ = (1f + _e1882_);
                                let _e1889_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1883_)));
                                let _e1892_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1884_)));
                                let _e1895_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1885_)));
                                let _e1898_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1886_)));
                                let _e1908_ = (_e1889_ + (_e1889_ * (1f - (_e1883_ * _e1889_))));
                                let _e1910_ = (_e1892_ + (_e1892_ * (1f - (_e1884_ * _e1892_))));
                                let _e1912_ = (_e1895_ + (_e1895_ * (1f - (_e1885_ * _e1895_))));
                                let _e1914_ = (_e1898_ + (_e1898_ * (1f - (_e1886_ * _e1898_))));
                                let _e1940_ = (1f - (_e543_ * select((_e1908_ + (_e1908_ * (1f - (_e1883_ * _e1908_)))), 1f, (_e1879_ <= 0f))));
                                let _e1942_ = (1f - (_e543_ * select((_e1910_ + (_e1910_ * (1f - (_e1884_ * _e1910_)))), 1f, (_e1880_ <= 0f))));
                                let _e1944_ = (1f - (_e543_ * select((_e1912_ + (_e1912_ * (1f - (_e1885_ * _e1912_)))), 1f, (_e1881_ <= 0f))));
                                let _e1946_ = (1f - (_e543_ * select((_e1914_ + (_e1914_ * (1f - (_e1886_ * _e1914_)))), 1f, (_e1882_ <= 0f))));
                                if (_e1940_ != _e1940_) {
                                    phi_8077_ = true;
                                } else {
                                    phi_8077_ = (0f >= _e1940_);
                                }
                                let _e1950_ = phi_8077_;
                                let _e1951_ = select(_e1940_, 0f, _e1950_);
                                if (_e1942_ != _e1942_) {
                                    phi_8092_ = true;
                                } else {
                                    phi_8092_ = (0f >= _e1942_);
                                }
                                let _e1955_ = phi_8092_;
                                let _e1956_ = select(_e1942_, 0f, _e1955_);
                                if (_e1944_ != _e1944_) {
                                    phi_8107_ = true;
                                } else {
                                    phi_8107_ = (0f >= _e1944_);
                                }
                                let _e1960_ = phi_8107_;
                                let _e1961_ = select(_e1944_, 0f, _e1960_);
                                if (_e1946_ != _e1946_) {
                                    phi_8122_ = true;
                                } else {
                                    phi_8122_ = (0f >= _e1946_);
                                }
                                let _e1965_ = phi_8122_;
                                let _e1966_ = select(_e1946_, 0f, _e1965_);
                                if (_e1951_ != _e1951_) {
                                    phi_8137_ = true;
                                } else {
                                    phi_8137_ = (1f <= _e1951_);
                                }
                                let _e1970_ = phi_8137_;
                                let _e1971_ = select(_e1951_, 1f, _e1970_);
                                if (_e1956_ != _e1956_) {
                                    phi_8152_ = true;
                                } else {
                                    phi_8152_ = (1f <= _e1956_);
                                }
                                let _e1975_ = phi_8152_;
                                let _e1976_ = select(_e1956_, 1f, _e1975_);
                                if (_e1961_ != _e1961_) {
                                    phi_8167_ = true;
                                } else {
                                    phi_8167_ = (1f <= _e1961_);
                                }
                                let _e1980_ = phi_8167_;
                                let _e1981_ = select(_e1961_, 1f, _e1980_);
                                if (_e1966_ != _e1966_) {
                                    phi_8182_ = true;
                                } else {
                                    phi_8182_ = (1f <= _e1966_);
                                }
                                let _e1985_ = phi_8182_;
                                let _e1986_ = select(_e1966_, 1f, _e1985_);
                                let _e1989_ = (_e537_ * _e1350_.member[0]);
                                let _e1992_ = (_e537_ * _e1350_.member[1]);
                                let _e1995_ = (_e537_ * _e1350_.member[2]);
                                let _e1998_ = (_e537_ * _e1350_.member[3]);
                                let _e2000_ = (_e1388_ - (_e1572_ * _e1989_));
                                let _e2002_ = (_e1388_ - (_e1572_ * _e1992_));
                                let _e2004_ = (_e1388_ - (_e1572_ * _e1995_));
                                let _e2006_ = (_e1388_ - (_e1572_ * _e1998_));
                                if (_e2000_ != _e2000_) {
                                    phi_8197_ = true;
                                } else {
                                    phi_8197_ = (-1f >= _e2000_);
                                }
                                let _e2010_ = phi_8197_;
                                let _e2011_ = select(_e2000_, -1f, _e2010_);
                                if (_e2002_ != _e2002_) {
                                    phi_8212_ = true;
                                } else {
                                    phi_8212_ = (-1f >= _e2002_);
                                }
                                let _e2015_ = phi_8212_;
                                let _e2016_ = select(_e2002_, -1f, _e2015_);
                                if (_e2004_ != _e2004_) {
                                    phi_8227_ = true;
                                } else {
                                    phi_8227_ = (-1f >= _e2004_);
                                }
                                let _e2020_ = phi_8227_;
                                let _e2021_ = select(_e2004_, -1f, _e2020_);
                                if (_e2006_ != _e2006_) {
                                    phi_8242_ = true;
                                } else {
                                    phi_8242_ = (-1f >= _e2006_);
                                }
                                let _e2025_ = phi_8242_;
                                let _e2026_ = select(_e2006_, -1f, _e2025_);
                                if (_e2011_ != _e2011_) {
                                    phi_8257_ = true;
                                } else {
                                    phi_8257_ = (1f <= _e2011_);
                                }
                                let _e2030_ = phi_8257_;
                                let _e2031_ = select(_e2011_, 1f, _e2030_);
                                if (_e2016_ != _e2016_) {
                                    phi_8272_ = true;
                                } else {
                                    phi_8272_ = (1f <= _e2016_);
                                }
                                let _e2035_ = phi_8272_;
                                let _e2036_ = select(_e2016_, 1f, _e2035_);
                                if (_e2021_ != _e2021_) {
                                    phi_8287_ = true;
                                } else {
                                    phi_8287_ = (1f <= _e2021_);
                                }
                                let _e2040_ = phi_8287_;
                                let _e2041_ = select(_e2021_, 1f, _e2040_);
                                if (_e2026_ != _e2026_) {
                                    phi_8302_ = true;
                                } else {
                                    phi_8302_ = (1f <= _e2026_);
                                }
                                let _e2045_ = phi_8302_;
                                let _e2046_ = select(_e2026_, 1f, _e2045_);
                                let _e2076_ = select(0f, _e2000_, (bitcast<f32>((bitcast<u32>(_e2000_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2078_ = select(0f, _e2002_, (bitcast<f32>((bitcast<u32>(_e2002_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2080_ = select(0f, _e2004_, (bitcast<f32>((bitcast<u32>(_e2004_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2082_ = select(0f, _e2006_, (bitcast<f32>((bitcast<u32>(_e2006_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                                let _e2083_ = (_e2076_ * 499.99997f);
                                let _e2084_ = (_e2078_ * 499.99997f);
                                let _e2085_ = (_e2080_ * 499.99997f);
                                let _e2086_ = (_e2082_ * 499.99997f);
                                if (_e2083_ != _e2083_) {
                                    phi_8317_ = true;
                                } else {
                                    phi_8317_ = (0f >= _e2083_);
                                }
                                let _e2090_ = phi_8317_;
                                let _e2091_ = select(_e2083_, 0f, _e2090_);
                                if (_e2084_ != _e2084_) {
                                    phi_8332_ = true;
                                } else {
                                    phi_8332_ = (0f >= _e2084_);
                                }
                                let _e2095_ = phi_8332_;
                                let _e2096_ = select(_e2084_, 0f, _e2095_);
                                if (_e2085_ != _e2085_) {
                                    phi_8347_ = true;
                                } else {
                                    phi_8347_ = (0f >= _e2085_);
                                }
                                let _e2100_ = phi_8347_;
                                let _e2101_ = select(_e2085_, 0f, _e2100_);
                                if (_e2086_ != _e2086_) {
                                    phi_8362_ = true;
                                } else {
                                    phi_8362_ = (0f >= _e2086_);
                                }
                                let _e2105_ = phi_8362_;
                                let _e2106_ = select(_e2086_, 0f, _e2105_);
                                if (_e2091_ != _e2091_) {
                                    phi_8377_ = true;
                                } else {
                                    phi_8377_ = (1f <= _e2091_);
                                }
                                let _e2110_ = phi_8377_;
                                let _e2111_ = select(_e2091_, 1f, _e2110_);
                                if (_e2096_ != _e2096_) {
                                    phi_8392_ = true;
                                } else {
                                    phi_8392_ = (1f <= _e2096_);
                                }
                                let _e2115_ = phi_8392_;
                                let _e2116_ = select(_e2096_, 1f, _e2115_);
                                if (_e2101_ != _e2101_) {
                                    phi_8407_ = true;
                                } else {
                                    phi_8407_ = (1f <= _e2101_);
                                }
                                let _e2120_ = phi_8407_;
                                let _e2121_ = select(_e2101_, 1f, _e2120_);
                                if (_e2106_ != _e2106_) {
                                    phi_8422_ = true;
                                } else {
                                    phi_8422_ = (1f <= _e2106_);
                                }
                                let _e2125_ = phi_8422_;
                                let _e2126_ = select(_e2106_, 1f, _e2125_);
                                let _e2147_ = (_e2076_ - 0.002f);
                                let _e2148_ = (_e2078_ - 0.002f);
                                let _e2149_ = (_e2080_ - 0.002f);
                                let _e2150_ = (_e2082_ - 0.002f);
                                if (_e2147_ != _e2147_) {
                                    phi_8437_ = true;
                                } else {
                                    phi_8437_ = (0f >= _e2147_);
                                }
                                let _e2154_ = phi_8437_;
                                if (_e2148_ != _e2148_) {
                                    phi_8452_ = true;
                                } else {
                                    phi_8452_ = (0f >= _e2148_);
                                }
                                let _e2159_ = phi_8452_;
                                if (_e2149_ != _e2149_) {
                                    phi_8467_ = true;
                                } else {
                                    phi_8467_ = (0f >= _e2149_);
                                }
                                let _e2164_ = phi_8467_;
                                if (_e2150_ != _e2150_) {
                                    phi_8482_ = true;
                                } else {
                                    phi_8482_ = (0f >= _e2150_);
                                }
                                let _e2169_ = phi_8482_;
                                let _e2171_ = ((0.001f * (((_e2111_ * _e2111_) * _e2111_) * (2f - _e2111_))) + select(_e2147_, 0f, _e2154_));
                                let _e2172_ = ((0.001f * (((_e2116_ * _e2116_) * _e2116_) * (2f - _e2116_))) + select(_e2148_, 0f, _e2159_));
                                let _e2173_ = ((0.001f * (((_e2121_ * _e2121_) * _e2121_) * (2f - _e2121_))) + select(_e2149_, 0f, _e2164_));
                                let _e2174_ = ((0.001f * (((_e2126_ * _e2126_) * _e2126_) * (2f - _e2126_))) + select(_e2150_, 0f, _e2169_));
                                let _e2211_ = bitcast<u32>((((1f - (_e561_ * _e1971_)) * _e1989_) + (_e1971_ * select(((_e1571_ * _e2171_) + (_e561_ * (_e2031_ * (1f - (0.33333334f * (_e2031_ * _e2031_)))))), _e2171_, _e1761_))));
                                let _e2212_ = (_e2211_ & 2139095040u);
                                if (_e2212_ == 2139095040u) {
                                    phi_5168_ = true;
                                } else {
                                    phi_5168_ = (_e2212_ == 0u);
                                }
                                let _e2216_ = phi_5168_;
                                let _e2218_ = bitcast<f32>(select(_e2211_, 0u, _e2216_));
                                let _e2219_ = bitcast<u32>((((1f - (_e561_ * _e1976_)) * _e1992_) + (_e1976_ * select(((_e1571_ * _e2172_) + (_e561_ * (_e2036_ * (1f - (0.33333334f * (_e2036_ * _e2036_)))))), _e2172_, _e1761_))));
                                let _e2220_ = (_e2219_ & 2139095040u);
                                if (_e2220_ == 2139095040u) {
                                    phi_5184_ = true;
                                } else {
                                    phi_5184_ = (_e2220_ == 0u);
                                }
                                let _e2224_ = phi_5184_;
                                let _e2227_ = bitcast<u32>((((1f - (_e561_ * _e1981_)) * _e1995_) + (_e1981_ * select(((_e1571_ * _e2173_) + (_e561_ * (_e2041_ * (1f - (0.33333334f * (_e2041_ * _e2041_)))))), _e2173_, _e1761_))));
                                let _e2228_ = (_e2227_ & 2139095040u);
                                if (_e2228_ == 2139095040u) {
                                    phi_5200_ = true;
                                } else {
                                    phi_5200_ = (_e2228_ == 0u);
                                }
                                let _e2232_ = phi_5200_;
                                let _e2235_ = bitcast<u32>((((1f - (_e561_ * _e1986_)) * _e1998_) + (_e1986_ * select(((_e1571_ * _e2174_) + (_e561_ * (_e2046_ * (1f - (0.33333334f * (_e2046_ * _e2046_)))))), _e2174_, _e1761_))));
                                let _e2236_ = (_e2235_ & 2139095040u);
                                if (_e2236_ == 2139095040u) {
                                    phi_5216_ = true;
                                } else {
                                    phi_5216_ = (_e2236_ == 0u);
                                }
                                let _e2240_ = phi_5216_;
                                let _e2246_ = (0.5f * (_e1793_ + _e2218_));
                                let _e2253_ = (1024u + _e1367_.member_1_);
                                if (_e2253_ < 1536u) {
                                } else {
                                    phi_10548_ = true;
                                    break;
                                }
                                scan_scratch[_e2253_] = (_e1793_ + (_e567_ * (_e2246_ - _e1793_)));
                                let _e2256_ = (1280u + _e1367_.member_1_);
                                if (_e2256_ < 1536u) {
                                } else {
                                    phi_10548_ = true;
                                    break;
                                }
                                scan_scratch[_e2256_] = (_e2218_ + (_e567_ * (_e2246_ - _e2218_)));
                                phi_4111_ = type_23_(array<f32, 4>(_e2218_, bitcast<f32>(select(_e2219_, 0u, _e2224_)), bitcast<f32>(select(_e2227_, 0u, _e2232_)), bitcast<f32>(select(_e2235_, 0u, _e2240_))));
                                phi_4113_ = type_23_(array<f32, 4>(_e1793_, bitcast<f32>(select(_e1794_, 0u, _e1799_)), bitcast<f32>(select(_e1802_, 0u, _e1807_)), bitcast<f32>(select(_e1810_, 0u, _e1815_))));
                            } else {
                                phi_4111_ = type_23_();
                                phi_4113_ = type_23_();
                            }
                            let _e2260_ = phi_4111_;
                            let _e2262_ = phi_4113_;
                            phi_4107_ = _e1365_;
                            phi_4110_ = _e2260_;
                            phi_4112_ = _e2262_;
                            phi_10548_ = _e1303_;
                            if !(_e1371_) {
                                break;
                            }
                        }
                        let _e2265_ = phi_10548_;
                        phi_10753_ = _e2265_;
                        if _e2265_ {
                            break;
                        }
                        phi_10976_ = _e2265_;
                    } else {
                        phi_10976_ = _e1303_;
                    }
                    let _e2267_ = phi_10976_;
                    workgroupBarrier();
                    if _e157_ {
                        if _e882_ {
                        } else {
                            phi_10753_ = true;
                            break;
                        }
                        let _e2268_ = scan_scratch[_e881_];
                        phi_5304_ = _e2268_;
                    } else {
                        phi_5304_ = 0f;
                    }
                    let _e2270_ = phi_5304_;
                    if _e157_ {
                        if _e886_ {
                        } else {
                            phi_10753_ = true;
                            break;
                        }
                        let _e2271_ = scan_scratch[_e885_];
                        phi_5316_ = _e2271_;
                    } else {
                        phi_5316_ = 0f;
                    }
                    let _e2273_ = phi_5316_;
                    let _e2275_ = (1f - (_e772_ * _e2270_));
                    if (_e2275_ != _e2275_) {
                        phi_8497_ = true;
                    } else {
                        phi_8497_ = (0.05f >= _e2275_);
                    }
                    let _e2279_ = phi_8497_;
                    let _e2282_ = (1f - (_e772_ * _e2273_));
                    if (_e2282_ != _e2282_) {
                        phi_8512_ = true;
                    } else {
                        phi_8512_ = (0.05f >= _e2282_);
                    }
                    let _e2286_ = phi_8512_;
                    if _e157_ {
                        let _e2291_ = (((_e1105_.member * _e811_) * select(_e2275_, 0.05f, _e2279_)) + (_e850_ * _e2270_));
                        let _e2296_ = select(0f, _e2291_, (bitcast<f32>((bitcast<u32>(_e2291_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                        if (_e2296_ != _e2296_) {
                            phi_8527_ = true;
                        } else {
                            phi_8527_ = (-4096f >= _e2296_);
                        }
                        let _e2300_ = phi_8527_;
                        let _e2301_ = select(_e2296_, -4096f, _e2300_);
                        if (_e2301_ != _e2301_) {
                            phi_8587_ = true;
                        } else {
                            phi_8587_ = (4096f <= _e2301_);
                        }
                        let _e2305_ = phi_8587_;
                        phi_5372_ = select(_e2301_, 4096f, _e2305_);
                    } else {
                        phi_5372_ = 0f;
                    }
                    let _e2308_ = phi_5372_;
                    if _e157_ {
                        let _e2312_ = (((_e1342_.member * _e811_) * select(_e2282_, 0.05f, _e2286_)) + (_e850_ * _e2273_));
                        let _e2317_ = select(0f, _e2312_, (bitcast<f32>((bitcast<u32>(_e2312_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                        if (_e2317_ != _e2317_) {
                            phi_8647_ = true;
                        } else {
                            phi_8647_ = (-4096f >= _e2317_);
                        }
                        let _e2321_ = phi_8647_;
                        let _e2322_ = select(_e2317_, -4096f, _e2321_);
                        if (_e2322_ != _e2322_) {
                            phi_8707_ = true;
                        } else {
                            phi_8707_ = (4096f <= _e2322_);
                        }
                        let _e2326_ = phi_8707_;
                        phi_5422_ = select(_e2322_, 4096f, _e2326_);
                    } else {
                        phi_5422_ = 0f;
                    }
                    let _e2329_ = phi_5422_;
                    workgroupBarrier();
                    if _e866_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e864_] = _e2308_;
                    if _e870_ {
                    } else {
                        phi_10753_ = true;
                        break;
                    }
                    scan_scratch[_e869_] = _e2329_;
                    workgroupBarrier();
                    let _e2330_ = (_e111_ > 0u);
                    if _e2330_ {
                        if _e157_ {
                            let _e2331_ = (_e111_ - 1u);
                            if (_e2331_ < 1536u) {
                            } else {
                                phi_10753_ = true;
                                break;
                            }
                            let _e2334_ = scan_scratch[_e2331_];
                            phi_5451_ = _e2334_;
                        } else {
                            phi_5451_ = f32();
                        }
                        let _e2336_ = phi_5451_;
                        phi_5453_ = _e2336_;
                        phi_5454_ = select(true, false, _e157_);
                    } else {
                        phi_5453_ = f32();
                        phi_5454_ = true;
                    }
                    let _e2339_ = phi_5453_;
                    let _e2341_ = phi_5454_;
                    let _e2342_ = select(_e2339_, 0f, _e2341_);
                    if _e2330_ {
                        if _e157_ {
                            let _e2343_ = (_e111_ + 255u);
                            if (_e2343_ < 1536u) {
                            } else {
                                phi_10753_ = true;
                                break;
                            }
                            let _e2346_ = scan_scratch[_e2343_];
                            phi_5475_ = _e2346_;
                        } else {
                            phi_5475_ = f32();
                        }
                        let _e2348_ = phi_5475_;
                        phi_5477_ = _e2348_;
                        phi_5478_ = select(true, false, _e157_);
                    } else {
                        phi_5477_ = f32();
                        phi_5478_ = true;
                    }
                    let _e2351_ = phi_5477_;
                    let _e2353_ = phi_5478_;
                    let _e2354_ = select(_e2351_, 0f, _e2353_);
                    let _e2359_ = select(0f, _e2342_, (bitcast<f32>((bitcast<u32>(_e2342_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2359_ != _e2359_) {
                        phi_8767_ = true;
                    } else {
                        phi_8767_ = (-4096f >= _e2359_);
                    }
                    let _e2363_ = phi_8767_;
                    let _e2364_ = select(_e2359_, -4096f, _e2363_);
                    if (_e2364_ != _e2364_) {
                        phi_8827_ = true;
                    } else {
                        phi_8827_ = (4096f <= _e2364_);
                    }
                    let _e2368_ = phi_8827_;
                    let _e2369_ = select(_e2364_, 4096f, _e2368_);
                    let _e2374_ = select(0f, _e2308_, (bitcast<f32>((bitcast<u32>(_e2308_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2374_ != _e2374_) {
                        phi_8887_ = true;
                    } else {
                        phi_8887_ = (-4096f >= _e2374_);
                    }
                    let _e2378_ = phi_8887_;
                    let _e2379_ = select(_e2374_, -4096f, _e2378_);
                    if (_e2379_ != _e2379_) {
                        phi_8947_ = true;
                    } else {
                        phi_8947_ = (4096f <= _e2379_);
                    }
                    let _e2383_ = phi_8947_;
                    let _e2384_ = select(_e2379_, 4096f, _e2383_);
                    let _e2387_ = (1f + (_e2384_ * _e2384_));
                    let _e2388_ = (1f + (_e2369_ * _e2369_));
                    if (_e2387_ != _e2387_) {
                        phi_9007_ = true;
                    } else {
                        phi_9007_ = (0.000000000001f >= _e2387_);
                    }
                    let _e2392_ = phi_9007_;
                    let _e2393_ = select(_e2387_, 0.000000000001f, _e2392_);
                    let _e2394_ = (_e2393_ * 0.5f);
                    let _e2399_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2393_) >> bitcast<u32>(1i))));
                    let _e2403_ = (_e2399_ * (1.5f - (_e2394_ * (_e2399_ * _e2399_))));
                    let _e2407_ = (_e2403_ * (1.5f - (_e2394_ * (_e2403_ * _e2403_))));
                    let _e2411_ = (_e2407_ * (1.5f - (_e2394_ * (_e2407_ * _e2407_))));
                    if (_e2388_ != _e2388_) {
                        phi_9067_ = true;
                    } else {
                        phi_9067_ = (0.000000000001f >= _e2388_);
                    }
                    let _e2415_ = phi_9067_;
                    let _e2416_ = select(_e2388_, 0.000000000001f, _e2415_);
                    let _e2417_ = (_e2416_ * 0.5f);
                    let _e2422_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2416_) >> bitcast<u32>(1i))));
                    let _e2426_ = (_e2422_ * (1.5f - (_e2417_ * (_e2422_ * _e2422_))));
                    let _e2430_ = (_e2426_ * (1.5f - (_e2417_ * (_e2426_ * _e2426_))));
                    let _e2434_ = (_e2430_ * (1.5f - (_e2417_ * (_e2430_ * _e2430_))));
                    let _e2437_ = ((_e2387_ * _e2411_) + (_e2388_ * _e2434_));
                    let _e2440_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2437_)));
                    let _e2444_ = (_e2440_ + (_e2440_ * (1f - (_e2437_ * _e2440_))));
                    let _e2448_ = (_e2444_ + (_e2444_ * (1f - (_e2437_ * _e2444_))));
                    let _e2458_ = (((_e2384_ + _e2369_) * (_e2448_ + (_e2448_ * (1f - (_e2437_ * _e2448_))))) * (1f + (0.5f * (_e2411_ * _e2434_))));
                    let _e2461_ = bitcast<f32>((bitcast<u32>(_e2369_) & 2147483647u));
                    let _e2464_ = bitcast<f32>((bitcast<u32>(_e2384_) & 2147483647u));
                    if (_e2461_ != _e2461_) {
                        phi_9127_ = true;
                    } else {
                        phi_9127_ = (_e2464_ >= _e2461_);
                    }
                    let _e2468_ = phi_9127_;
                    if (_e2458_ != _e2458_) {
                        phi_9187_ = true;
                    } else {
                        phi_9187_ = (-1f >= _e2458_);
                    }
                    let _e2473_ = phi_9187_;
                    let _e2474_ = select(_e2458_, -1f, _e2473_);
                    if (_e2474_ != _e2474_) {
                        phi_9247_ = true;
                    } else {
                        phi_9247_ = (1f <= _e2474_);
                    }
                    let _e2478_ = phi_9247_;
                    let _e2486_ = select(0f, _e2354_, (bitcast<f32>((bitcast<u32>(_e2354_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2486_ != _e2486_) {
                        phi_9427_ = true;
                    } else {
                        phi_9427_ = (-4096f >= _e2486_);
                    }
                    let _e2490_ = phi_9427_;
                    let _e2491_ = select(_e2486_, -4096f, _e2490_);
                    if (_e2491_ != _e2491_) {
                        phi_9487_ = true;
                    } else {
                        phi_9487_ = (4096f <= _e2491_);
                    }
                    let _e2495_ = phi_9487_;
                    let _e2496_ = select(_e2491_, 4096f, _e2495_);
                    let _e2501_ = select(0f, _e2329_, (bitcast<f32>((bitcast<u32>(_e2329_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    if (_e2501_ != _e2501_) {
                        phi_9547_ = true;
                    } else {
                        phi_9547_ = (-4096f >= _e2501_);
                    }
                    let _e2505_ = phi_9547_;
                    let _e2506_ = select(_e2501_, -4096f, _e2505_);
                    if (_e2506_ != _e2506_) {
                        phi_9607_ = true;
                    } else {
                        phi_9607_ = (4096f <= _e2506_);
                    }
                    let _e2510_ = phi_9607_;
                    let _e2511_ = select(_e2506_, 4096f, _e2510_);
                    let _e2514_ = (1f + (_e2511_ * _e2511_));
                    let _e2515_ = (1f + (_e2496_ * _e2496_));
                    if (_e2514_ != _e2514_) {
                        phi_9667_ = true;
                    } else {
                        phi_9667_ = (0.000000000001f >= _e2514_);
                    }
                    let _e2519_ = phi_9667_;
                    let _e2520_ = select(_e2514_, 0.000000000001f, _e2519_);
                    let _e2521_ = (_e2520_ * 0.5f);
                    let _e2526_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2520_) >> bitcast<u32>(1i))));
                    let _e2530_ = (_e2526_ * (1.5f - (_e2521_ * (_e2526_ * _e2526_))));
                    let _e2534_ = (_e2530_ * (1.5f - (_e2521_ * (_e2530_ * _e2530_))));
                    let _e2538_ = (_e2534_ * (1.5f - (_e2521_ * (_e2534_ * _e2534_))));
                    if (_e2515_ != _e2515_) {
                        phi_9727_ = true;
                    } else {
                        phi_9727_ = (0.000000000001f >= _e2515_);
                    }
                    let _e2542_ = phi_9727_;
                    let _e2543_ = select(_e2515_, 0.000000000001f, _e2542_);
                    let _e2544_ = (_e2543_ * 0.5f);
                    let _e2549_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2543_) >> bitcast<u32>(1i))));
                    let _e2553_ = (_e2549_ * (1.5f - (_e2544_ * (_e2549_ * _e2549_))));
                    let _e2557_ = (_e2553_ * (1.5f - (_e2544_ * (_e2553_ * _e2553_))));
                    let _e2561_ = (_e2557_ * (1.5f - (_e2544_ * (_e2557_ * _e2557_))));
                    let _e2564_ = ((_e2514_ * _e2538_) + (_e2515_ * _e2561_));
                    let _e2567_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2564_)));
                    let _e2571_ = (_e2567_ + (_e2567_ * (1f - (_e2564_ * _e2567_))));
                    let _e2575_ = (_e2571_ + (_e2571_ * (1f - (_e2564_ * _e2571_))));
                    let _e2585_ = (((_e2511_ + _e2496_) * (_e2575_ + (_e2575_ * (1f - (_e2564_ * _e2575_))))) * (1f + (0.5f * (_e2538_ * _e2561_))));
                    let _e2588_ = bitcast<f32>((bitcast<u32>(_e2496_) & 2147483647u));
                    let _e2591_ = bitcast<f32>((bitcast<u32>(_e2511_) & 2147483647u));
                    if (_e2588_ != _e2588_) {
                        phi_9787_ = true;
                    } else {
                        phi_9787_ = (_e2591_ >= _e2588_);
                    }
                    let _e2595_ = phi_9787_;
                    if (_e2585_ != _e2585_) {
                        phi_9847_ = true;
                    } else {
                        phi_9847_ = (-1f >= _e2585_);
                    }
                    let _e2600_ = phi_9847_;
                    let _e2601_ = select(_e2585_, -1f, _e2600_);
                    if (_e2601_ != _e2601_) {
                        phi_9907_ = true;
                    } else {
                        phi_9907_ = (1f <= _e2601_);
                    }
                    let _e2605_ = phi_9907_;
                    if _e157_ {
                        phi_6756_ = (_e200_ + select(select(_e2601_, 1f, _e2605_), 0f, (select(_e2588_, _e2591_, _e2595_) <= 0f)));
                        phi_6757_ = (_e202_ + select(select(_e2474_, 1f, _e2478_), 0f, (select(_e2461_, _e2464_, _e2468_) <= 0f)));
                    } else {
                        phi_6756_ = _e200_;
                        phi_6757_ = _e202_;
                    }
                    let _e2612_ = phi_6756_;
                    let _e2614_ = phi_6757_;
                    workgroupBarrier();
                    phi_10754_ = _e2267_;
                    phi_305_ = _e2612_;
                    phi_307_ = _e2614_;
                } else {
                    phi_10754_ = _e196_;
                    phi_305_ = f32();
                    phi_307_ = f32();
                }
                let _e2616_ = phi_10754_;
                let _e2618_ = phi_305_;
                let _e2620_ = phi_307_;
                phi_10393_ = _e2616_;
                phi_301_ = _e215_;
                phi_304_ = _e2618_;
                phi_306_ = _e2620_;
                phi_10753_ = _e2616_;
                if !(_e221_) {
                    break;
                }
            }
            let _e2623_ = phi_10753_;
            if _e2623_ {
                break;
            }
            if _e157_ {
                if (_e116_ == 0u) {
                    phi_6771_ = 0u;
                } else {
                    let _e2625_ = (_e116_ - 1u);
                    phi_6771_ = select(_e2625_, _e159_, (_e159_ < _e2625_));
                }
                let _e2629_ = phi_6771_;
                if (_e2629_ < _e116_) {
                } else {
                    break;
                }
                let _e2634_ = target_audio.member[_e2629_].x;
                let _e2638_ = target_audio.member[_e2629_].y;
                let _e2639_ = bitcast<u32>(_e2634_);
                let _e2640_ = (_e2639_ & 2139095040u);
                if (_e2640_ == 2139095040u) {
                    phi_6794_ = true;
                } else {
                    phi_6794_ = (_e2640_ == 0u);
                }
                let _e2644_ = phi_6794_;
                let _e2647_ = bitcast<u32>(_e2638_);
                let _e2648_ = (_e2647_ & 2139095040u);
                if (_e2648_ == 2139095040u) {
                    phi_6810_ = true;
                } else {
                    phi_6810_ = (_e2648_ == 0u);
                }
                let _e2652_ = phi_6810_;
                phi_6818_ = vec2<f32>(bitcast<f32>(select(_e2639_, 0u, _e2644_)), bitcast<f32>(select(_e2647_, 0u, _e2652_)));
            } else {
                phi_6818_ = vec2<f32>(0f, 0f);
            }
            let _e2657_ = phi_6818_;
            if _e157_ {
                let _e2660_ = local;
                phi_6824_ = (_e2660_ - _e2657_.x);
            } else {
                phi_6824_ = 0f;
            }
            let _e2663_ = phi_6824_;
            if _e157_ {
                let _e2666_ = local_1_;
                phi_6830_ = (_e2666_ - _e2657_.y);
            } else {
                phi_6830_ = 0f;
            }
            let _e2669_ = phi_6830_;
            if _e157_ {
                let _e2673_ = bitcast<u32>(((_e2663_ * _e2663_) + (_e2669_ * _e2669_)));
                let _e2674_ = (_e2673_ & 2139095040u);
                if (_e2674_ == 2139095040u) {
                    phi_6847_ = true;
                } else {
                    phi_6847_ = (_e2674_ == 0u);
                }
                let _e2678_ = phi_6847_;
                phi_6853_ = bitcast<f32>(select(_e2673_, 0u, _e2678_));
            } else {
                phi_6853_ = 0f;
            }
            let _e2682_ = phi_6853_;
            if _e157_ {
                let _e2688_ = bitcast<u32>(((_e2657_.x * _e2657_.x) + (_e2657_.y * _e2657_.y)));
                let _e2689_ = (_e2688_ & 2139095040u);
                if (_e2689_ == 2139095040u) {
                    phi_6874_ = true;
                } else {
                    phi_6874_ = (_e2689_ == 0u);
                }
                let _e2693_ = phi_6874_;
                phi_6880_ = bitcast<f32>(select(_e2688_, 0u, _e2693_));
            } else {
                phi_6880_ = 0f;
            }
            let _e2697_ = phi_6880_;
            let _e2698_ = bitcast<u32>(_e2682_);
            let _e2699_ = (_e2698_ & 2139095040u);
            if (_e2699_ == 2139095040u) {
                phi_6891_ = true;
            } else {
                phi_6891_ = (_e2699_ == 0u);
            }
            let _e2703_ = phi_6891_;
            let _e2706_ = bitcast<u32>(_e2697_);
            let _e2707_ = (_e2706_ & 2139095040u);
            if (_e2707_ == 2139095040u) {
                phi_6907_ = true;
            } else {
                phi_6907_ = (_e2707_ == 0u);
            }
            let _e2711_ = phi_6907_;
            let _e4223 = subgroupAdd(bitcast<f32>(select(_e2698_, 0u, _e2703_)));
            let _e4227 = subgroupAdd(bitcast<f32>(select(_e2706_, 0u, _e2711_)));
            let _e2716_ = (_e112_ == 0u);
            let _e2720_ = select(255u, _e111_, (_e111_ < 255u));
            let _e2721_ = (_e2720_ < 256u);
            if _e2721_ {
            } else {
                break;
            }
            err_scratch[_e2720_] = select(0f, _e4223, _e2716_);
            if _e2721_ {
            } else {
                break;
            }
            tgt_scratch[_e2720_] = select(0f, _e4227, _e2716_);
            workgroupBarrier();
            phi_6945_ = 128u;
            loop {
                let _e2725_ = phi_6945_;
                let _e2726_ = (_e2725_ > 0u);
                if _e2726_ {
                    if (_e111_ < _e2725_) {
                        let _e2728_ = (_e111_ + _e2725_);
                        let _e2729_ = (_e2728_ < 256u);
                        if _e2729_ {
                        } else {
                            phi_10989_ = true;
                            break;
                        }
                        let _e2731_ = err_scratch[_e2728_];
                        let _e2732_ = (_e111_ < 256u);
                        if _e2732_ {
                        } else {
                            phi_10989_ = true;
                            break;
                        }
                        let _e2734_ = err_scratch[_e111_];
                        err_scratch[_e111_] = (_e2734_ + _e2731_);
                        if _e2729_ {
                        } else {
                            phi_10989_ = true;
                            break;
                        }
                        let _e2737_ = tgt_scratch[_e2728_];
                        if _e2732_ {
                        } else {
                            phi_10989_ = true;
                            break;
                        }
                        let _e2739_ = tgt_scratch[_e111_];
                        tgt_scratch[_e111_] = (_e2739_ + _e2737_);
                    }
                    workgroupBarrier();
                    phi_6946_ = (_e2725_ >> bitcast<u32>(1i));
                } else {
                    phi_6946_ = u32();
                }
                let _e2744_ = phi_6946_;
                phi_6945_ = _e2744_;
                phi_10989_ = _e2623_;
                if !(_e2726_) {
                    break;
                }
            }
            let _e2747_ = phi_10989_;
            if _e2747_ {
                break;
            }
            let _e2749_ = err_scratch[0];
            let _e2750_ = bitcast<u32>(_e2749_);
            let _e2751_ = (_e2750_ & 2139095040u);
            if (_e2751_ == 2139095040u) {
                phi_7006_ = true;
            } else {
                phi_7006_ = (_e2751_ == 0u);
            }
            let _e2755_ = phi_7006_;
            let _e2759_ = tgt_scratch[0];
            let _e2760_ = bitcast<u32>(_e2759_);
            let _e2761_ = (_e2760_ & 2139095040u);
            if (_e2761_ == 2139095040u) {
                phi_7024_ = true;
            } else {
                phi_7024_ = (_e2761_ == 0u);
            }
            let _e2765_ = phi_7024_;
            workgroupBarrier();
            if (_e111_ == 0u) {
                if (_e110_.x < _e128_) {
                    if (_e110_.y < _e148_) {
                        if (_e110_.z < select(1u, _e139_, (1u < _e139_))) {
                            let _e2774_ = uniforms.member.member_5_;
                            if ((bitcast<u32>(_e2774_) & 2139095040u) != 2139095040u) {
                                let _e2778_ = (_e2774_ > 0.000000000001f);
                                phi_7064_ = select(f32(), _e2774_, _e2778_);
                                phi_7065_ = select(true, false, _e2778_);
                            } else {
                                phi_7064_ = f32();
                                phi_7065_ = true;
                            }
                            let _e2782_ = phi_7064_;
                            let _e2784_ = phi_7065_;
                            let _e2786_ = (bitcast<f32>(select(_e2760_, 0u, _e2765_)) + select(_e2782_, 0.000000000001f, _e2784_));
                            if ((bitcast<u32>(_e2786_) & 2139095040u) != 2139095040u) {
                                let _e2790_ = (_e2786_ <= 0f);
                                if _e2790_ {
                                    phi_7163_ = f32();
                                } else {
                                    if (_e2786_ < 0.00001f) {
                                        let _e2822_ = (select(0.000000000000000000000001f, _e2786_, (_e2786_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                        if (_e2822_ != _e2822_) {
                                            phi_10147_ = true;
                                        } else {
                                            phi_10147_ = (0.000001f >= _e2822_);
                                        }
                                        let _e2826_ = phi_10147_;
                                        let _e2827_ = select(_e2822_, 0.000001f, _e2826_);
                                        let _e2830_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2827_)));
                                        let _e2834_ = (_e2830_ + (_e2830_ * (1f - (_e2827_ * _e2830_))));
                                        let _e2838_ = (_e2834_ + (_e2834_ * (1f - (_e2827_ * _e2834_))));
                                        let _e2844_ = bitcast<u32>(((_e2838_ + (_e2838_ * (1f - (_e2827_ * _e2838_)))) * 18446744000000000000f));
                                        let _e2845_ = (_e2844_ & 2139095040u);
                                        if (_e2845_ == 2139095040u) {
                                            phi_7121_ = true;
                                        } else {
                                            phi_7121_ = (_e2845_ == 0u);
                                        }
                                        let _e2849_ = phi_7121_;
                                        phi_7162_ = bitcast<f32>(select(_e2844_, 0u, _e2849_));
                                    } else {
                                        if (_e2786_ != _e2786_) {
                                            phi_10087_ = true;
                                        } else {
                                            phi_10087_ = (0.000001f >= _e2786_);
                                        }
                                        let _e2795_ = phi_10087_;
                                        let _e2796_ = select(_e2786_, 0.000001f, _e2795_);
                                        let _e2799_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2796_)));
                                        let _e2803_ = (_e2799_ + (_e2799_ * (1f - (_e2796_ * _e2799_))));
                                        let _e2807_ = (_e2803_ + (_e2803_ * (1f - (_e2796_ * _e2803_))));
                                        let _e2812_ = bitcast<u32>((_e2807_ + (_e2807_ * (1f - (_e2796_ * _e2807_)))));
                                        let _e2813_ = (_e2812_ & 2139095040u);
                                        if (_e2813_ == 2139095040u) {
                                            phi_7156_ = true;
                                        } else {
                                            phi_7156_ = (_e2813_ == 0u);
                                        }
                                        let _e2817_ = phi_7156_;
                                        phi_7162_ = bitcast<f32>(select(_e2812_, 0u, _e2817_));
                                    }
                                    let _e2853_ = phi_7162_;
                                    phi_7163_ = _e2853_;
                                }
                                let _e2855_ = phi_7163_;
                                phi_7164_ = _e2855_;
                                phi_7165_ = _e2790_;
                            } else {
                                phi_7164_ = f32();
                                phi_7165_ = true;
                            }
                            let _e2857_ = phi_7164_;
                            let _e2859_ = phi_7165_;
                            let _e2862_ = bitcast<u32>((bitcast<f32>(select(_e2750_, 0u, _e2755_)) * select(_e2857_, 0f, _e2859_)));
                            let _e2863_ = (_e2862_ & 2139095040u);
                            if (_e2863_ == 2139095040u) {
                                phi_7181_ = true;
                            } else {
                                phi_7181_ = (_e2863_ == 0u);
                            }
                            let _e2867_ = phi_7181_;
                            let _e2873_ = ((((_e110_.z * _e128_) + _e110_.x) * _e148_) + _e110_.y);
                            if (_e120_ == 0u) {
                                phi_7201_ = 0u;
                            } else {
                                let _e2875_ = (_e120_ - 1u);
                                phi_7201_ = select(_e2875_, _e2873_, (_e2873_ < _e2875_));
                            }
                            let _e2879_ = phi_7201_;
                            if (_e2879_ < _e120_) {
                            } else {
                                break;
                            }
                            seed_losses.member[_e2879_] = bitcast<f32>(select(_e2862_, 0u, _e2867_));
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
