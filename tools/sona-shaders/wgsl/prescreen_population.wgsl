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
    var phi_222_: u32;
    var phi_229_: u32;
    var phi_252_: u32;
    var phi_275_: bool;
    var phi_291_: bool;
    var phi_299_: vec2<f32>;
    var phi_11972_: bool;
    var phi_302_: type_15_;
    var phi_305_: f32;
    var phi_307_: f32;
    var phi_303_: type_15_;
    var phi_330_: type_15_;
    var phi_348_: u32;
    var phi_359_: u32;
    var phi_371_: u32;
    var phi_390_: u32;
    var phi_423_: f32;
    var phi_429_: f32;
    var phi_8096_: bool;
    var phi_600_: bool;
    var phi_606_: f32;
    var phi_607_: f32;
    var phi_526_: f32;
    var phi_528_: f32;
    var phi_529_: bool;
    var phi_549_: f32;
    var phi_608_: f32;
    var phi_609_: f32;
    var phi_8156_: bool;
    var phi_695_: bool;
    var phi_8216_: bool;
    var phi_660_: bool;
    var phi_701_: f32;
    var phi_702_: f32;
    var phi_703_: f32;
    var phi_704_: bool;
    var phi_720_: u32;
    var phi_738_: bool;
    var phi_755_: u32;
    var phi_773_: bool;
    var phi_790_: u32;
    var phi_808_: bool;
    var phi_825_: u32;
    var phi_843_: bool;
    var phi_860_: u32;
    var phi_878_: bool;
    var phi_895_: u32;
    var phi_913_: bool;
    var phi_8276_: bool;
    var phi_8336_: bool;
    var phi_1494_: bool;
    var phi_1500_: f32;
    var phi_1501_: f32;
    var phi_1420_: f32;
    var phi_1422_: f32;
    var phi_1423_: bool;
    var phi_1502_: f32;
    var phi_1514_: u32;
    var phi_1532_: bool;
    var phi_1549_: u32;
    var phi_1567_: bool;
    var phi_1584_: u32;
    var phi_1602_: bool;
    var phi_1619_: u32;
    var phi_1637_: bool;
    var phi_1654_: u32;
    var phi_1778_: bool;
    var phi_1784_: f32;
    var phi_1785_: f32;
    var phi_1704_: f32;
    var phi_1706_: f32;
    var phi_1707_: bool;
    var phi_1727_: f32;
    var phi_1786_: f32;
    var phi_1798_: u32;
    var phi_1922_: bool;
    var phi_1928_: f32;
    var phi_1929_: f32;
    var phi_1848_: f32;
    var phi_1850_: f32;
    var phi_1851_: bool;
    var phi_1871_: f32;
    var phi_1930_: f32;
    var phi_1942_: u32;
    var phi_2066_: bool;
    var phi_2072_: f32;
    var phi_2073_: f32;
    var phi_1992_: f32;
    var phi_1994_: f32;
    var phi_1995_: bool;
    var phi_2015_: f32;
    var phi_2074_: f32;
    var phi_2086_: u32;
    var phi_2210_: bool;
    var phi_2216_: f32;
    var phi_2217_: f32;
    var phi_2136_: f32;
    var phi_2138_: f32;
    var phi_2139_: bool;
    var phi_2159_: f32;
    var phi_2218_: f32;
    var phi_2240_: type_23_;
    var phi_2300_: u32;
    var phi_2301_: u32;
    var phi_11776_: bool;
    var phi_2563_: u32;
    var phi_2564_: u32;
    var phi_12077_: bool;
    var phi_3106_: bool;
    var phi_3122_: bool;
    var phi_3130_: type_20_;
    var phi_3153_: type_23_;
    var phi_3213_: u32;
    var phi_3214_: u32;
    var phi_12092_: bool;
    var phi_3476_: u32;
    var phi_3477_: u32;
    var phi_12112_: bool;
    var phi_4019_: bool;
    var phi_4035_: bool;
    var phi_4043_: type_20_;
    var phi_4108_: type_15_;
    var phi_4111_: type_22_;
    var phi_4113_: type_22_;
    var phi_4109_: type_15_;
    var phi_4136_: type_15_;
    var phi_4190_: u32;
    var phi_4191_: u32;
    var phi_4204_: u32;
    var phi_4205_: u32;
    var phi_4218_: u32;
    var phi_4219_: u32;
    var phi_4232_: u32;
    var phi_4233_: u32;
    var phi_8396_: bool;
    var phi_8411_: bool;
    var phi_8426_: bool;
    var phi_8441_: bool;
    var phi_8456_: bool;
    var phi_8471_: bool;
    var phi_8486_: bool;
    var phi_8501_: bool;
    var phi_8516_: bool;
    var phi_8531_: bool;
    var phi_8546_: bool;
    var phi_8561_: bool;
    var phi_8576_: bool;
    var phi_8591_: bool;
    var phi_8606_: bool;
    var phi_8621_: bool;
    var phi_8636_: bool;
    var phi_8651_: bool;
    var phi_8666_: bool;
    var phi_8681_: bool;
    var phi_8696_: bool;
    var phi_8711_: bool;
    var phi_8726_: bool;
    var phi_8741_: bool;
    var phi_8756_: bool;
    var phi_8771_: bool;
    var phi_8786_: bool;
    var phi_8801_: bool;
    var phi_8816_: bool;
    var phi_8831_: bool;
    var phi_8846_: bool;
    var phi_8861_: bool;
    var phi_4938_: u32;
    var phi_4939_: u32;
    var phi_4952_: u32;
    var phi_4953_: u32;
    var phi_4966_: u32;
    var phi_4967_: u32;
    var phi_4980_: u32;
    var phi_4981_: u32;
    var phi_5000_: u32;
    var phi_5001_: u32;
    var phi_5014_: u32;
    var phi_5015_: u32;
    var phi_5028_: u32;
    var phi_5029_: u32;
    var phi_5042_: u32;
    var phi_5043_: u32;
    var phi_8996_: bool;
    var phi_9011_: bool;
    var phi_9026_: bool;
    var phi_9041_: bool;
    var phi_9056_: bool;
    var phi_9071_: bool;
    var phi_9086_: bool;
    var phi_9101_: bool;
    var phi_9116_: bool;
    var phi_9131_: bool;
    var phi_9146_: bool;
    var phi_9161_: bool;
    var phi_9176_: bool;
    var phi_9191_: bool;
    var phi_9206_: bool;
    var phi_9221_: bool;
    var phi_9236_: bool;
    var phi_9251_: bool;
    var phi_9266_: bool;
    var phi_9281_: bool;
    var phi_9296_: bool;
    var phi_9311_: bool;
    var phi_9326_: bool;
    var phi_9341_: bool;
    var phi_9356_: bool;
    var phi_9371_: bool;
    var phi_9386_: bool;
    var phi_9401_: bool;
    var phi_9416_: bool;
    var phi_9431_: bool;
    var phi_9446_: bool;
    var phi_9461_: bool;
    var phi_5748_: u32;
    var phi_5749_: u32;
    var phi_5762_: u32;
    var phi_5763_: u32;
    var phi_5776_: u32;
    var phi_5777_: u32;
    var phi_5790_: u32;
    var phi_5791_: u32;
    var phi_4112_: type_22_;
    var phi_4114_: type_22_;
    var phi_12127_: bool;
    var phi_12809_: bool;
    var phi_5875_: f32;
    var phi_5887_: f32;
    var phi_9596_: bool;
    var phi_9611_: bool;
    var phi_5911_: bool;
    var phi_5917_: f32;
    var phi_5935_: bool;
    var phi_5941_: f32;
    var phi_5970_: f32;
    var phi_5972_: f32;
    var phi_5973_: bool;
    var phi_5994_: f32;
    var phi_5996_: f32;
    var phi_5997_: bool;
    var phi_9626_: bool;
    var phi_9641_: bool;
    var phi_9656_: bool;
    var phi_9671_: bool;
    var phi_9686_: bool;
    var phi_9701_: bool;
    var phi_9716_: bool;
    var phi_9731_: bool;
    var phi_9746_: bool;
    var phi_9761_: bool;
    var phi_9776_: bool;
    var phi_9791_: bool;
    var phi_9806_: bool;
    var phi_9821_: bool;
    var phi_9836_: bool;
    var phi_9851_: bool;
    var phi_9866_: bool;
    var phi_9881_: bool;
    var phi_9896_: bool;
    var phi_9911_: bool;
    var phi_9926_: bool;
    var phi_9941_: bool;
    var phi_9956_: bool;
    var phi_9971_: bool;
    var phi_9986_: bool;
    var phi_10001_: bool;
    var phi_10016_: bool;
    var phi_10031_: bool;
    var phi_10046_: bool;
    var phi_10061_: bool;
    var phi_10076_: bool;
    var phi_10091_: bool;
    var phi_10106_: bool;
    var phi_10121_: bool;
    var phi_10136_: bool;
    var phi_10151_: bool;
    var phi_10166_: bool;
    var phi_10181_: bool;
    var phi_10196_: bool;
    var phi_10211_: bool;
    var phi_10226_: bool;
    var phi_10241_: bool;
    var phi_10256_: bool;
    var phi_10271_: bool;
    var phi_10286_: bool;
    var phi_10301_: bool;
    var phi_10316_: bool;
    var phi_10331_: bool;
    var phi_10346_: bool;
    var phi_10361_: bool;
    var phi_10376_: bool;
    var phi_10391_: bool;
    var phi_10406_: bool;
    var phi_10421_: bool;
    var phi_10436_: bool;
    var phi_10451_: bool;
    var phi_10466_: bool;
    var phi_10481_: bool;
    var phi_10496_: bool;
    var phi_10511_: bool;
    var phi_10526_: bool;
    var phi_10541_: bool;
    var phi_10556_: bool;
    var phi_10571_: bool;
    var phi_6424_: bool;
    var phi_6426_: bool;
    var phi_6427_: bool;
    var phi_6428_: bool;
    var phi_6429_: bool;
    var phi_10586_: bool;
    var phi_10601_: bool;
    var phi_10616_: bool;
    var phi_10631_: bool;
    var phi_6807_: type_27_;
    var phi_10646_: bool;
    var phi_10661_: bool;
    var phi_10676_: bool;
    var phi_10691_: bool;
    var phi_10706_: bool;
    var phi_10721_: bool;
    var phi_10736_: bool;
    var phi_10751_: bool;
    var phi_10766_: bool;
    var phi_10781_: bool;
    var phi_10796_: bool;
    var phi_10811_: bool;
    var phi_10826_: bool;
    var phi_10841_: bool;
    var phi_10856_: bool;
    var phi_10871_: bool;
    var phi_10886_: bool;
    var phi_10901_: bool;
    var phi_10916_: bool;
    var phi_10931_: bool;
    var phi_10946_: bool;
    var phi_10961_: bool;
    var phi_10976_: bool;
    var phi_10991_: bool;
    var phi_11006_: bool;
    var phi_11021_: bool;
    var phi_11036_: bool;
    var phi_11051_: bool;
    var phi_11066_: bool;
    var phi_11081_: bool;
    var phi_11096_: bool;
    var phi_11111_: bool;
    var phi_11126_: bool;
    var phi_11141_: bool;
    var phi_11156_: bool;
    var phi_11171_: bool;
    var phi_11186_: bool;
    var phi_11201_: bool;
    var phi_11216_: bool;
    var phi_11231_: bool;
    var phi_11246_: bool;
    var phi_11261_: bool;
    var phi_11276_: bool;
    var phi_11291_: bool;
    var phi_11306_: bool;
    var phi_11321_: bool;
    var phi_11336_: bool;
    var phi_11351_: bool;
    var phi_11366_: bool;
    var phi_11381_: bool;
    var phi_11396_: bool;
    var phi_11411_: bool;
    var phi_11426_: bool;
    var phi_11441_: bool;
    var phi_11456_: bool;
    var phi_11471_: bool;
    var phi_11486_: bool;
    var phi_11501_: bool;
    var phi_11516_: bool;
    var phi_11531_: bool;
    var phi_11546_: bool;
    var phi_11561_: bool;
    var phi_11576_: bool;
    var phi_11591_: bool;
    var phi_7231_: bool;
    var phi_7233_: bool;
    var phi_7234_: bool;
    var phi_7235_: bool;
    var phi_7236_: bool;
    var phi_11606_: bool;
    var phi_11621_: bool;
    var phi_11636_: bool;
    var phi_11651_: bool;
    var phi_7614_: type_27_;
    var phi_7623_: f32;
    var phi_7624_: f32;
    var phi_12413_: bool;
    var phi_306_: f32;
    var phi_308_: f32;
    var phi_12412_: bool;
    var phi_7638_: u32;
    var phi_7661_: bool;
    var phi_7677_: bool;
    var phi_7685_: vec2<f32>;
    var local: f32;
    var phi_7691_: f32;
    var local_1_: f32;
    var phi_7697_: f32;
    var phi_7714_: bool;
    var phi_7720_: f32;
    var phi_7741_: bool;
    var phi_7747_: f32;
    var phi_7758_: bool;
    var phi_7774_: bool;
    var phi_7804_: u32;
    var phi_7805_: u32;
    var phi_12820_: bool;
    var phi_7865_: bool;
    var phi_7883_: bool;
    var phi_7923_: f32;
    var phi_7924_: bool;
    var phi_11666_: bool;
    var phi_8015_: bool;
    var phi_11726_: bool;
    var phi_7980_: bool;
    var phi_8021_: f32;
    var phi_8022_: f32;
    var phi_8023_: f32;
    var phi_8024_: bool;
    var phi_8040_: bool;
    var phi_8060_: u32;

    switch bitcast<i32>(0u) {
        default: {
            let _e112_ = wg_id_1_;
            let _e113_ = lid_1_;
            let _e115_ = arrayLength((&input_audio.member));
            let _e117_ = arrayLength((&target_audio.member));
            let _e119_ = arrayLength((&seed_params.member));
            let _e121_ = arrayLength((&seed_losses.member));
            let _e127_ = uniforms.member.member;
            let _e129_ = select(1u, _e127_, (1u < _e127_));
            let _e132_ = uniforms.member.member_1_;
            let _e135_ = uniforms.member.member_2_;
            let _e137_ = select(1u, _e135_, (1u < _e135_));
            let _e140_ = uniforms.member.member_3_;
            if (_e132_ == 0u) {
                phi_222_ = 1u;
            } else {
                phi_222_ = (((_e132_ - 1u) >> bitcast<u32>(8i)) + 1u);
            }
            let _e149_ = phi_222_;
            let _e150_ = (_e112_.y * 256u);
            if (_e132_ > _e150_) {
                phi_229_ = (_e132_ - _e150_);
            } else {
                phi_229_ = 0u;
            }
            let _e154_ = phi_229_;
            let _e157_ = (_e150_ + _e113_);
            let _e158_ = (_e157_ < _e132_);
            let _e160_ = ((_e112_.z * _e132_) + _e157_);
            if _e158_ {
                if (_e115_ == 0u) {
                    phi_252_ = 0u;
                } else {
                    let _e162_ = (_e115_ - 1u);
                    phi_252_ = select(_e162_, _e160_, (_e160_ < _e162_));
                }
                let _e166_ = phi_252_;
                if (_e166_ < _e115_) {
                } else {
                    break;
                }
                let _e171_ = input_audio.member[_e166_].x;
                let _e175_ = input_audio.member[_e166_].y;
                let _e176_ = bitcast<u32>(_e171_);
                let _e177_ = (_e176_ & 2139095040u);
                if (_e177_ == 2139095040u) {
                    phi_275_ = true;
                } else {
                    phi_275_ = (_e177_ == 0u);
                }
                let _e181_ = phi_275_;
                let _e184_ = bitcast<u32>(_e175_);
                let _e185_ = (_e184_ & 2139095040u);
                if (_e185_ == 2139095040u) {
                    phi_291_ = true;
                } else {
                    phi_291_ = (_e185_ == 0u);
                }
                let _e189_ = phi_291_;
                phi_299_ = vec2<f32>(bitcast<f32>(select(_e176_, 0u, _e181_)), bitcast<f32>(select(_e184_, 0u, _e189_)));
            } else {
                phi_299_ = vec2<f32>(0f, 0f);
            }
            let _e194_ = phi_299_;
            phi_11972_ = false;
            phi_302_ = type_15_(0u, _e137_);
            phi_305_ = 0f;
            phi_307_ = 0f;
            loop {
                let _e197_ = phi_11972_;
                let _e199_ = phi_302_;
                let _e201_ = phi_305_;
                let _e203_ = phi_307_;
                local = _e203_;
                local_1_ = _e201_;
                if (_e199_.member < _e199_.member_1_) {
                    phi_303_ = type_15_((_e199_.member + 1u), _e199_.member_1_);
                    phi_330_ = type_15_(1u, _e199_.member);
                } else {
                    phi_303_ = _e199_;
                    phi_330_ = type_15_(0u, u32());
                }
                let _e216_ = phi_303_;
                let _e218_ = phi_330_;
                let _e222_ = (bitcast<i32>(_e218_.member) != 0i);
                if _e222_ {
                    if (_e129_ == 0u) {
                        phi_348_ = 0u;
                    } else {
                        let _e224_ = (_e129_ - 1u);
                        phi_348_ = select(_e224_, _e112_.x, (_e112_.x < _e224_));
                    }
                    let _e228_ = phi_348_;
                    if (_e119_ >= ((_e129_ * _e137_) * 16u)) {
                        phi_359_ = (((_e228_ * _e137_) + _e218_.member_1_) * 16u);
                    } else {
                        phi_359_ = (_e228_ * 16u);
                    }
                    let _e237_ = phi_359_;
                    let _e238_ = (_e119_ == 0u);
                    if _e238_ {
                        phi_371_ = 0u;
                    } else {
                        let _e239_ = (_e119_ - 1u);
                        phi_371_ = select(_e239_, _e237_, (_e237_ < _e239_));
                    }
                    let _e243_ = phi_371_;
                    if (_e243_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e247_ = seed_params.member[_e243_];
                    let _e248_ = (_e237_ + 1u);
                    if _e238_ {
                        phi_390_ = 0u;
                    } else {
                        let _e249_ = (_e119_ - 1u);
                        phi_390_ = select(_e249_, _e248_, (_e248_ < _e249_));
                    }
                    let _e253_ = phi_390_;
                    if (_e253_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e257_ = seed_params.member[_e253_];
                    let _e261_ = select(0.001f, _e247_, ((bitcast<u32>(_e247_) & 2139095040u) != 2139095040u));
                    if (_e261_ != _e261_) {
                        phi_423_ = 0.00001f;
                    } else {
                        let _e264_ = select(_e261_, 0.00001f, (_e261_ < 0.00001f));
                        phi_423_ = select(_e264_, 0.495f, (_e264_ > 0.495f));
                    }
                    let _e268_ = phi_423_;
                    let _e269_ = (_e268_ <= 0.25f);
                    if _e269_ {
                        phi_429_ = _e268_;
                    } else {
                        phi_429_ = (0.5f - _e268_);
                    }
                    let _e272_ = phi_429_;
                    let _e273_ = (3.1415927f * _e272_);
                    let _e274_ = (_e273_ * _e273_);
                    let _e285_ = (_e273_ * (1f + (_e274_ * (-0.16666667f + (_e274_ * (0.008333334f + (_e274_ * (-0.0001984127f + (_e274_ * (0.0000027557319f + (_e274_ * -0.000000025052108f)))))))))));
                    let _e295_ = (1f + (_e274_ * (-0.5f + (_e274_ * (0.041666668f + (_e274_ * (-0.0013888889f + (_e274_ * (0.000024801588f + (_e274_ * -0.0000002755732f))))))))));
                    let _e297_ = select(_e285_, _e295_, _e269_);
                    if (_e297_ != _e297_) {
                        phi_8096_ = true;
                    } else {
                        phi_8096_ = (0.000001f >= _e297_);
                    }
                    let _e301_ = phi_8096_;
                    let _e302_ = select(_e297_, 0.000001f, _e301_);
                    let _e305_ = bitcast<f32>((2129859011u - bitcast<u32>(_e302_)));
                    let _e309_ = (_e305_ + (_e305_ * (1f - (_e302_ * _e305_))));
                    let _e313_ = (_e309_ + (_e309_ * (1f - (_e302_ * _e309_))));
                    let _e318_ = (select(_e295_, _e285_, _e269_) * (_e313_ + (_e313_ * (1f - (_e302_ * _e313_)))));
                    let _e319_ = bitcast<u32>(_e257_);
                    let _e320_ = (_e319_ & 2139095040u);
                    if (_e320_ != 2139095040u) {
                        let _e322_ = (_e320_ == 2139095040u);
                        if _e322_ {
                            if ((_e319_ & 2147483647u) > 2139095040u) {
                                phi_528_ = f32();
                                phi_529_ = true;
                            } else {
                                let _e337_ = ((_e319_ & 2147483648u) == 0u);
                                if _e337_ {
                                    phi_526_ = 64f;
                                } else {
                                    phi_526_ = f32();
                                }
                                let _e339_ = phi_526_;
                                phi_528_ = _e339_;
                                phi_529_ = select(true, false, _e337_);
                            }
                            let _e342_ = phi_528_;
                            let _e344_ = phi_529_;
                            if _e344_ {
                                phi_549_ = 0.05f;
                            } else {
                                phi_549_ = _e342_;
                            }
                            let _e346_ = phi_549_;
                            phi_608_ = _e346_;
                        } else {
                            if (_e257_ < 0.05f) {
                                phi_607_ = 0.05f;
                            } else {
                                if (_e257_ > 64f) {
                                    phi_606_ = 64f;
                                } else {
                                    if _e322_ {
                                        phi_600_ = true;
                                    } else {
                                        phi_600_ = (_e320_ == 0u);
                                    }
                                    let _e327_ = phi_600_;
                                    phi_606_ = bitcast<f32>(select(_e319_, 0u, _e327_));
                                }
                                let _e331_ = phi_606_;
                                phi_607_ = _e331_;
                            }
                            let _e333_ = phi_607_;
                            phi_608_ = _e333_;
                        }
                        let _e348_ = phi_608_;
                        phi_609_ = _e348_;
                    } else {
                        phi_609_ = 0.70710677f;
                    }
                    let _e350_ = phi_609_;
                    if ((bitcast<u32>(_e350_) & 2139095040u) != 2139095040u) {
                        let _e354_ = (_e350_ <= 0f);
                        if _e354_ {
                            phi_702_ = f32();
                        } else {
                            if (_e350_ < 0.00001f) {
                                let _e386_ = (select(0.000000000000000000000001f, _e350_, (_e350_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                if (_e386_ != _e386_) {
                                    phi_8216_ = true;
                                } else {
                                    phi_8216_ = (0.000001f >= _e386_);
                                }
                                let _e390_ = phi_8216_;
                                let _e391_ = select(_e386_, 0.000001f, _e390_);
                                let _e394_ = bitcast<f32>((2129859011u - bitcast<u32>(_e391_)));
                                let _e398_ = (_e394_ + (_e394_ * (1f - (_e391_ * _e394_))));
                                let _e402_ = (_e398_ + (_e398_ * (1f - (_e391_ * _e398_))));
                                let _e408_ = bitcast<u32>(((_e402_ + (_e402_ * (1f - (_e391_ * _e402_)))) * 18446744000000000000f));
                                let _e409_ = (_e408_ & 2139095040u);
                                if (_e409_ == 2139095040u) {
                                    phi_660_ = true;
                                } else {
                                    phi_660_ = (_e409_ == 0u);
                                }
                                let _e413_ = phi_660_;
                                phi_701_ = bitcast<f32>(select(_e408_, 0u, _e413_));
                            } else {
                                if (_e350_ != _e350_) {
                                    phi_8156_ = true;
                                } else {
                                    phi_8156_ = (0.000001f >= _e350_);
                                }
                                let _e359_ = phi_8156_;
                                let _e360_ = select(_e350_, 0.000001f, _e359_);
                                let _e363_ = bitcast<f32>((2129859011u - bitcast<u32>(_e360_)));
                                let _e367_ = (_e363_ + (_e363_ * (1f - (_e360_ * _e363_))));
                                let _e371_ = (_e367_ + (_e367_ * (1f - (_e360_ * _e367_))));
                                let _e376_ = bitcast<u32>((_e371_ + (_e371_ * (1f - (_e360_ * _e371_)))));
                                let _e377_ = (_e376_ & 2139095040u);
                                if (_e377_ == 2139095040u) {
                                    phi_695_ = true;
                                } else {
                                    phi_695_ = (_e377_ == 0u);
                                }
                                let _e381_ = phi_695_;
                                phi_701_ = bitcast<f32>(select(_e376_, 0u, _e381_));
                            }
                            let _e417_ = phi_701_;
                            phi_702_ = _e417_;
                        }
                        let _e419_ = phi_702_;
                        phi_703_ = _e419_;
                        phi_704_ = _e354_;
                    } else {
                        phi_703_ = f32();
                        phi_704_ = true;
                    }
                    let _e421_ = phi_703_;
                    let _e423_ = phi_704_;
                    let _e424_ = select(_e421_, 0f, _e423_);
                    let _e425_ = (_e237_ + 6u);
                    if _e238_ {
                        phi_720_ = 0u;
                    } else {
                        let _e426_ = (_e119_ - 1u);
                        phi_720_ = select(_e426_, _e425_, (_e425_ < _e426_));
                    }
                    let _e430_ = phi_720_;
                    if (_e430_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e434_ = seed_params.member[_e430_];
                    let _e435_ = bitcast<u32>(_e434_);
                    let _e436_ = (_e435_ & 2139095040u);
                    if (_e436_ == 2139095040u) {
                        phi_738_ = true;
                    } else {
                        phi_738_ = (_e436_ == 0u);
                    }
                    let _e440_ = phi_738_;
                    let _e442_ = bitcast<f32>(select(_e435_, 0u, _e440_));
                    let _e443_ = (_e237_ + 7u);
                    if _e238_ {
                        phi_755_ = 0u;
                    } else {
                        let _e444_ = (_e119_ - 1u);
                        phi_755_ = select(_e444_, _e443_, (_e443_ < _e444_));
                    }
                    let _e448_ = phi_755_;
                    if (_e448_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e452_ = seed_params.member[_e448_];
                    let _e453_ = bitcast<u32>(_e452_);
                    let _e454_ = (_e453_ & 2139095040u);
                    if (_e454_ == 2139095040u) {
                        phi_773_ = true;
                    } else {
                        phi_773_ = (_e454_ == 0u);
                    }
                    let _e458_ = phi_773_;
                    let _e460_ = bitcast<f32>(select(_e453_, 0u, _e458_));
                    let _e461_ = (_e237_ + 8u);
                    if _e238_ {
                        phi_790_ = 0u;
                    } else {
                        let _e462_ = (_e119_ - 1u);
                        phi_790_ = select(_e462_, _e461_, (_e461_ < _e462_));
                    }
                    let _e466_ = phi_790_;
                    if (_e466_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e470_ = seed_params.member[_e466_];
                    let _e471_ = bitcast<u32>(_e470_);
                    let _e472_ = (_e471_ & 2139095040u);
                    if (_e472_ == 2139095040u) {
                        phi_808_ = true;
                    } else {
                        phi_808_ = (_e472_ == 0u);
                    }
                    let _e476_ = phi_808_;
                    let _e478_ = bitcast<f32>(select(_e471_, 0u, _e476_));
                    let _e479_ = (_e237_ + 9u);
                    if _e238_ {
                        phi_825_ = 0u;
                    } else {
                        let _e480_ = (_e119_ - 1u);
                        phi_825_ = select(_e480_, _e479_, (_e479_ < _e480_));
                    }
                    let _e484_ = phi_825_;
                    if (_e484_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e488_ = seed_params.member[_e484_];
                    let _e489_ = bitcast<u32>(_e488_);
                    let _e490_ = (_e489_ & 2139095040u);
                    if (_e490_ == 2139095040u) {
                        phi_843_ = true;
                    } else {
                        phi_843_ = (_e490_ == 0u);
                    }
                    let _e494_ = phi_843_;
                    let _e496_ = bitcast<f32>(select(_e489_, 0u, _e494_));
                    let _e497_ = (_e237_ + 10u);
                    if _e238_ {
                        phi_860_ = 0u;
                    } else {
                        let _e498_ = (_e119_ - 1u);
                        phi_860_ = select(_e498_, _e497_, (_e497_ < _e498_));
                    }
                    let _e502_ = phi_860_;
                    if (_e502_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e506_ = seed_params.member[_e502_];
                    let _e507_ = bitcast<u32>(_e506_);
                    let _e508_ = (_e507_ & 2139095040u);
                    if (_e508_ == 2139095040u) {
                        phi_878_ = true;
                    } else {
                        phi_878_ = (_e508_ == 0u);
                    }
                    let _e512_ = phi_878_;
                    let _e514_ = bitcast<f32>(select(_e507_, 0u, _e512_));
                    let _e515_ = (_e237_ + 11u);
                    if _e238_ {
                        phi_895_ = 0u;
                    } else {
                        let _e516_ = (_e119_ - 1u);
                        phi_895_ = select(_e516_, _e515_, (_e515_ < _e516_));
                    }
                    let _e520_ = phi_895_;
                    if (_e520_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e524_ = seed_params.member[_e520_];
                    let _e525_ = bitcast<u32>(_e524_);
                    let _e526_ = (_e525_ & 2139095040u);
                    if (_e526_ == 2139095040u) {
                        phi_913_ = true;
                    } else {
                        phi_913_ = (_e526_ == 0u);
                    }
                    let _e530_ = phi_913_;
                    let _e532_ = bitcast<f32>(select(_e525_, 0u, _e530_));
                    let _e538_ = select(select(select(0f, 0.999995f, (0.999995f <= _e442_)), _e442_, (_e442_ <= 0.999995f)), 0f, (_e442_ <= 0f));
                    let _e556_ = select(select(select(0f, 64f, (64f <= _e496_)), _e496_, (_e496_ <= 64f)), 0f, (_e496_ <= 0f));
                    let _e562_ = select(select(select(0f, 1f, (1f <= _e514_)), _e514_, (_e514_ <= 1f)), 0f, (_e514_ <= 0f));
                    let _e568_ = select(select(select(0f, 1f, (1f <= _e532_)), _e532_, (_e532_ <= 1f)), 0f, (_e532_ <= 0f));
                    if (_e318_ != _e318_) {
                        phi_8276_ = true;
                    } else {
                        phi_8276_ = (0.00001f >= _e318_);
                    }
                    let _e572_ = phi_8276_;
                    let _e573_ = select(_e318_, 0.00001f, _e572_);
                    if (_e424_ != _e424_) {
                        phi_8336_ = true;
                    } else {
                        phi_8336_ = (0f >= _e424_);
                    }
                    let _e577_ = phi_8336_;
                    let _e578_ = select(_e424_, 0f, _e577_);
                    let _e581_ = (1f + (_e573_ * (_e573_ + _e578_)));
                    let _e584_ = bitcast<f32>((2129859011u - bitcast<u32>(_e581_)));
                    let _e588_ = (_e584_ + (_e584_ * (1f - (_e581_ * _e584_))));
                    let _e592_ = (_e588_ + (_e588_ * (1f - (_e581_ * _e588_))));
                    let _e593_ = (_e573_ * _e592_);
                    let _e594_ = (_e573_ * _e593_);
                    let _e595_ = (_e318_ + _e318_);
                    let _e596_ = bitcast<u32>(_e595_);
                    let _e597_ = (_e596_ & 2139095040u);
                    let _e598_ = (_e597_ == 2139095040u);
                    if _e598_ {
                        if ((_e596_ & 2147483647u) > 2139095040u) {
                            phi_1422_ = f32();
                            phi_1423_ = true;
                        } else {
                            let _e613_ = ((_e596_ & 2147483648u) == 0u);
                            if _e613_ {
                                phi_1420_ = 128f;
                            } else {
                                phi_1420_ = f32();
                            }
                            let _e615_ = phi_1420_;
                            phi_1422_ = _e615_;
                            phi_1423_ = select(true, false, _e613_);
                        }
                        let _e618_ = phi_1422_;
                        let _e620_ = phi_1423_;
                        phi_1502_ = select(_e618_, 0f, _e620_);
                    } else {
                        if (_e595_ < 0f) {
                            phi_1501_ = 0f;
                        } else {
                            if (_e595_ > 128f) {
                                phi_1500_ = 128f;
                            } else {
                                if _e598_ {
                                    phi_1494_ = true;
                                } else {
                                    phi_1494_ = (_e597_ == 0u);
                                }
                                let _e603_ = phi_1494_;
                                phi_1500_ = bitcast<f32>(select(_e596_, 0u, _e603_));
                            }
                            let _e607_ = phi_1500_;
                            phi_1501_ = _e607_;
                        }
                        let _e609_ = phi_1501_;
                        phi_1502_ = _e609_;
                    }
                    let _e623_ = phi_1502_;
                    let _e624_ = (_e237_ + 2u);
                    if _e238_ {
                        phi_1514_ = 0u;
                    } else {
                        let _e625_ = (_e119_ - 1u);
                        phi_1514_ = select(_e625_, _e624_, (_e624_ < _e625_));
                    }
                    let _e629_ = phi_1514_;
                    if (_e629_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e633_ = seed_params.member[_e629_];
                    let _e634_ = bitcast<u32>(_e633_);
                    let _e635_ = (_e634_ & 2139095040u);
                    if (_e635_ == 2139095040u) {
                        phi_1532_ = true;
                    } else {
                        phi_1532_ = (_e635_ == 0u);
                    }
                    let _e639_ = phi_1532_;
                    let _e641_ = bitcast<f32>(select(_e634_, 0u, _e639_));
                    let _e642_ = (_e237_ + 3u);
                    if _e238_ {
                        phi_1549_ = 0u;
                    } else {
                        let _e643_ = (_e119_ - 1u);
                        phi_1549_ = select(_e643_, _e642_, (_e642_ < _e643_));
                    }
                    let _e647_ = phi_1549_;
                    if (_e647_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e651_ = seed_params.member[_e647_];
                    let _e652_ = bitcast<u32>(_e651_);
                    let _e653_ = (_e652_ & 2139095040u);
                    if (_e653_ == 2139095040u) {
                        phi_1567_ = true;
                    } else {
                        phi_1567_ = (_e653_ == 0u);
                    }
                    let _e657_ = phi_1567_;
                    let _e659_ = bitcast<f32>(select(_e652_, 0u, _e657_));
                    let _e660_ = (_e237_ + 4u);
                    if _e238_ {
                        phi_1584_ = 0u;
                    } else {
                        let _e661_ = (_e119_ - 1u);
                        phi_1584_ = select(_e661_, _e660_, (_e660_ < _e661_));
                    }
                    let _e665_ = phi_1584_;
                    if (_e665_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e669_ = seed_params.member[_e665_];
                    let _e670_ = bitcast<u32>(_e669_);
                    let _e671_ = (_e670_ & 2139095040u);
                    if (_e671_ == 2139095040u) {
                        phi_1602_ = true;
                    } else {
                        phi_1602_ = (_e671_ == 0u);
                    }
                    let _e675_ = phi_1602_;
                    let _e677_ = bitcast<f32>(select(_e670_, 0u, _e675_));
                    let _e678_ = (_e237_ + 5u);
                    if _e238_ {
                        phi_1619_ = 0u;
                    } else {
                        let _e679_ = (_e119_ - 1u);
                        phi_1619_ = select(_e679_, _e678_, (_e678_ < _e679_));
                    }
                    let _e683_ = phi_1619_;
                    if (_e683_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e687_ = seed_params.member[_e683_];
                    let _e688_ = bitcast<u32>(_e687_);
                    let _e689_ = (_e688_ & 2139095040u);
                    if (_e689_ == 2139095040u) {
                        phi_1637_ = true;
                    } else {
                        phi_1637_ = (_e689_ == 0u);
                    }
                    let _e693_ = phi_1637_;
                    let _e695_ = bitcast<f32>(select(_e688_, 0u, _e693_));
                    let _e696_ = (_e237_ + 12u);
                    if _e238_ {
                        phi_1654_ = 0u;
                    } else {
                        let _e697_ = (_e119_ - 1u);
                        phi_1654_ = select(_e697_, _e696_, (_e696_ < _e697_));
                    }
                    let _e701_ = phi_1654_;
                    if (_e701_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e705_ = seed_params.member[_e701_];
                    let _e706_ = bitcast<u32>(_e705_);
                    let _e707_ = (_e706_ & 2139095040u);
                    let _e708_ = (_e707_ == 2139095040u);
                    if _e708_ {
                        if ((_e706_ & 2147483647u) > 2139095040u) {
                            phi_1706_ = f32();
                            phi_1707_ = true;
                        } else {
                            let _e723_ = ((_e706_ & 2147483648u) == 0u);
                            if _e723_ {
                                phi_1704_ = 1f;
                            } else {
                                phi_1704_ = f32();
                            }
                            let _e725_ = phi_1704_;
                            phi_1706_ = _e725_;
                            phi_1707_ = select(true, false, _e723_);
                        }
                        let _e728_ = phi_1706_;
                        let _e730_ = phi_1707_;
                        if _e730_ {
                            phi_1727_ = -1f;
                        } else {
                            phi_1727_ = _e728_;
                        }
                        let _e732_ = phi_1727_;
                        phi_1786_ = _e732_;
                    } else {
                        if (_e705_ < -1f) {
                            phi_1785_ = -1f;
                        } else {
                            if (_e705_ > 1f) {
                                phi_1784_ = 1f;
                            } else {
                                if _e708_ {
                                    phi_1778_ = true;
                                } else {
                                    phi_1778_ = (_e707_ == 0u);
                                }
                                let _e713_ = phi_1778_;
                                phi_1784_ = bitcast<f32>(select(_e706_, 0u, _e713_));
                            }
                            let _e717_ = phi_1784_;
                            phi_1785_ = _e717_;
                        }
                        let _e719_ = phi_1785_;
                        phi_1786_ = _e719_;
                    }
                    let _e734_ = phi_1786_;
                    let _e735_ = (_e237_ + 13u);
                    if _e238_ {
                        phi_1798_ = 0u;
                    } else {
                        let _e736_ = (_e119_ - 1u);
                        phi_1798_ = select(_e736_, _e735_, (_e735_ < _e736_));
                    }
                    let _e740_ = phi_1798_;
                    if (_e740_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e744_ = seed_params.member[_e740_];
                    let _e745_ = bitcast<u32>(_e744_);
                    let _e746_ = (_e745_ & 2139095040u);
                    let _e747_ = (_e746_ == 2139095040u);
                    if _e747_ {
                        if ((_e745_ & 2147483647u) > 2139095040u) {
                            phi_1850_ = f32();
                            phi_1851_ = true;
                        } else {
                            let _e762_ = ((_e745_ & 2147483648u) == 0u);
                            if _e762_ {
                                phi_1848_ = 4f;
                            } else {
                                phi_1848_ = f32();
                            }
                            let _e764_ = phi_1848_;
                            phi_1850_ = _e764_;
                            phi_1851_ = select(true, false, _e762_);
                        }
                        let _e767_ = phi_1850_;
                        let _e769_ = phi_1851_;
                        if _e769_ {
                            phi_1871_ = -4f;
                        } else {
                            phi_1871_ = _e767_;
                        }
                        let _e771_ = phi_1871_;
                        phi_1930_ = _e771_;
                    } else {
                        if (_e744_ < -4f) {
                            phi_1929_ = -4f;
                        } else {
                            if (_e744_ > 4f) {
                                phi_1928_ = 4f;
                            } else {
                                if _e747_ {
                                    phi_1922_ = true;
                                } else {
                                    phi_1922_ = (_e746_ == 0u);
                                }
                                let _e752_ = phi_1922_;
                                phi_1928_ = bitcast<f32>(select(_e745_, 0u, _e752_));
                            }
                            let _e756_ = phi_1928_;
                            phi_1929_ = _e756_;
                        }
                        let _e758_ = phi_1929_;
                        phi_1930_ = _e758_;
                    }
                    let _e773_ = phi_1930_;
                    let _e774_ = (_e237_ + 14u);
                    if _e238_ {
                        phi_1942_ = 0u;
                    } else {
                        let _e775_ = (_e119_ - 1u);
                        phi_1942_ = select(_e775_, _e774_, (_e774_ < _e775_));
                    }
                    let _e779_ = phi_1942_;
                    if (_e779_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e783_ = seed_params.member[_e779_];
                    let _e784_ = bitcast<u32>(_e783_);
                    let _e785_ = (_e784_ & 2139095040u);
                    let _e786_ = (_e785_ == 2139095040u);
                    if _e786_ {
                        if ((_e784_ & 2147483647u) > 2139095040u) {
                            phi_1994_ = f32();
                            phi_1995_ = true;
                        } else {
                            let _e801_ = ((_e784_ & 2147483648u) == 0u);
                            if _e801_ {
                                phi_1992_ = 64f;
                            } else {
                                phi_1992_ = f32();
                            }
                            let _e803_ = phi_1992_;
                            phi_1994_ = _e803_;
                            phi_1995_ = select(true, false, _e801_);
                        }
                        let _e806_ = phi_1994_;
                        let _e808_ = phi_1995_;
                        if _e808_ {
                            phi_2015_ = -64f;
                        } else {
                            phi_2015_ = _e806_;
                        }
                        let _e810_ = phi_2015_;
                        phi_2074_ = _e810_;
                    } else {
                        if (_e783_ < -64f) {
                            phi_2073_ = -64f;
                        } else {
                            if (_e783_ > 64f) {
                                phi_2072_ = 64f;
                            } else {
                                if _e786_ {
                                    phi_2066_ = true;
                                } else {
                                    phi_2066_ = (_e785_ == 0u);
                                }
                                let _e791_ = phi_2066_;
                                phi_2072_ = bitcast<f32>(select(_e784_, 0u, _e791_));
                            }
                            let _e795_ = phi_2072_;
                            phi_2073_ = _e795_;
                        }
                        let _e797_ = phi_2073_;
                        phi_2074_ = _e797_;
                    }
                    let _e812_ = phi_2074_;
                    let _e813_ = (_e237_ + 15u);
                    if _e238_ {
                        phi_2086_ = 0u;
                    } else {
                        let _e814_ = (_e119_ - 1u);
                        phi_2086_ = select(_e814_, _e813_, (_e813_ < _e814_));
                    }
                    let _e818_ = phi_2086_;
                    if (_e818_ < _e119_) {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e822_ = seed_params.member[_e818_];
                    let _e823_ = bitcast<u32>(_e822_);
                    let _e824_ = (_e823_ & 2139095040u);
                    let _e825_ = (_e824_ == 2139095040u);
                    if _e825_ {
                        if ((_e823_ & 2147483647u) > 2139095040u) {
                            phi_2138_ = f32();
                            phi_2139_ = true;
                        } else {
                            let _e840_ = ((_e823_ & 2147483648u) == 0u);
                            if _e840_ {
                                phi_2136_ = 64f;
                            } else {
                                phi_2136_ = f32();
                            }
                            let _e842_ = phi_2136_;
                            phi_2138_ = _e842_;
                            phi_2139_ = select(true, false, _e840_);
                        }
                        let _e845_ = phi_2138_;
                        let _e847_ = phi_2139_;
                        if _e847_ {
                            phi_2159_ = -64f;
                        } else {
                            phi_2159_ = _e845_;
                        }
                        let _e849_ = phi_2159_;
                        phi_2218_ = _e849_;
                    } else {
                        if (_e822_ < -64f) {
                            phi_2217_ = -64f;
                        } else {
                            if (_e822_ > 64f) {
                                phi_2216_ = 64f;
                            } else {
                                if _e825_ {
                                    phi_2210_ = true;
                                } else {
                                    phi_2210_ = (_e824_ == 0u);
                                }
                                let _e830_ = phi_2210_;
                                phi_2216_ = bitcast<f32>(select(_e823_, 0u, _e830_));
                            }
                            let _e834_ = phi_2216_;
                            phi_2217_ = _e834_;
                        }
                        let _e836_ = phi_2217_;
                        phi_2218_ = _e836_;
                    }
                    let _e851_ = phi_2218_;
                    if _e158_ {
                        let _e853_ = (_e593_ + _e593_);
                        let _e854_ = (_e594_ + _e594_);
                        phi_2240_ = type_23_(((_e592_ + _e592_) - 1f), -(_e853_), _e853_, (1f - _e854_), (_e853_ * _e194_.x), (_e854_ * _e194_.x));
                    } else {
                        phi_2240_ = type_23_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e863_ = phi_2240_;
                    let _e865_ = select(255u, _e113_, (_e113_ < 255u));
                    let _e867_ = (_e865_ < 1536u);
                    if _e867_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e863_.member;
                    let _e870_ = (256u + _e865_);
                    let _e871_ = (_e870_ < 1536u);
                    if _e871_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e863_.member_1_;
                    let _e874_ = (512u + _e865_);
                    let _e875_ = (_e874_ < 1536u);
                    if _e875_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e874_] = _e863_.member_2_;
                    let _e878_ = (768u + _e865_);
                    let _e879_ = (_e878_ < 1536u);
                    if _e879_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e878_] = _e863_.member_3_;
                    let _e882_ = (1024u + _e865_);
                    let _e883_ = (_e882_ < 1536u);
                    if _e883_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e882_] = _e863_.member_4_;
                    let _e886_ = (1280u + _e865_);
                    let _e887_ = (_e886_ < 1536u);
                    if _e887_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e886_] = _e863_.member_5_;
                    workgroupBarrier();
                    phi_2300_ = 1u;
                    loop {
                        let _e890_ = phi_2300_;
                        let _e891_ = (_e890_ < 256u);
                        if _e891_ {
                            let _e893_ = (_e890_ << bitcast<u32>(1i));
                            let _e896_ = (((_e113_ + 1u) * _e893_) - 1u);
                            if (_e896_ < 256u) {
                                let _e898_ = (_e896_ - _e890_);
                                let _e900_ = select(255u, _e898_, (_e898_ < 255u));
                                if (_e900_ < 1536u) {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e903_ = scan_scratch[_e900_];
                                let _e904_ = (256u + _e900_);
                                if (_e904_ < 1536u) {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e907_ = scan_scratch[_e904_];
                                let _e908_ = (512u + _e900_);
                                if (_e908_ < 1536u) {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e911_ = scan_scratch[_e908_];
                                let _e912_ = (768u + _e900_);
                                if (_e912_ < 1536u) {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e915_ = scan_scratch[_e912_];
                                let _e916_ = (1024u + _e900_);
                                if (_e916_ < 1536u) {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e919_ = scan_scratch[_e916_];
                                let _e920_ = (1280u + _e900_);
                                if (_e920_ < 1536u) {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e923_ = scan_scratch[_e920_];
                                let _e925_ = select(255u, _e896_, (_e896_ < 255u));
                                let _e926_ = (_e925_ < 1536u);
                                if _e926_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e928_ = scan_scratch[_e925_];
                                let _e929_ = (256u + _e925_);
                                let _e930_ = (_e929_ < 1536u);
                                if _e930_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e932_ = scan_scratch[_e929_];
                                let _e933_ = (512u + _e925_);
                                let _e934_ = (_e933_ < 1536u);
                                if _e934_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e936_ = scan_scratch[_e933_];
                                let _e937_ = (768u + _e925_);
                                let _e938_ = (_e937_ < 1536u);
                                if _e938_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e940_ = scan_scratch[_e937_];
                                let _e941_ = (1024u + _e925_);
                                let _e942_ = (_e941_ < 1536u);
                                if _e942_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e944_ = scan_scratch[_e941_];
                                let _e945_ = (1280u + _e925_);
                                let _e946_ = (_e945_ < 1536u);
                                if _e946_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                let _e948_ = scan_scratch[_e945_];
                                if _e926_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                scan_scratch[_e925_] = ((_e928_ * _e903_) + (_e932_ * _e911_));
                                if _e930_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                scan_scratch[_e929_] = ((_e928_ * _e907_) + (_e932_ * _e915_));
                                if _e934_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                scan_scratch[_e933_] = ((_e936_ * _e903_) + (_e940_ * _e911_));
                                if _e938_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                scan_scratch[_e937_] = ((_e936_ * _e907_) + (_e940_ * _e915_));
                                if _e942_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                scan_scratch[_e941_] = (((_e928_ * _e919_) + (_e932_ * _e923_)) + _e944_);
                                if _e946_ {
                                } else {
                                    phi_11776_ = true;
                                    break;
                                }
                                scan_scratch[_e945_] = (((_e936_ * _e919_) + (_e940_ * _e923_)) + _e948_);
                            }
                            workgroupBarrier();
                            phi_2301_ = _e893_;
                        } else {
                            phi_2301_ = u32();
                        }
                        let _e970_ = phi_2301_;
                        phi_2300_ = _e970_;
                        phi_11776_ = _e197_;
                        if !(_e891_) {
                            break;
                        }
                    }
                    let _e973_ = phi_11776_;
                    phi_12412_ = _e973_;
                    if _e973_ {
                        break;
                    }
                    let _e974_ = (_e113_ == 255u);
                    if _e974_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_2563_ = 128u;
                    loop {
                        let _e982_ = phi_2563_;
                        let _e983_ = (_e982_ > 0u);
                        if _e983_ {
                            let _e988_ = (((_e113_ + 1u) * (_e982_ << bitcast<u32>(1i))) - 1u);
                            if (_e988_ < 256u) {
                                let _e990_ = (_e988_ - _e982_);
                                let _e992_ = select(255u, _e990_, (_e990_ < 255u));
                                let _e993_ = (_e992_ < 1536u);
                                if _e993_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e995_ = scan_scratch[_e992_];
                                let _e996_ = (256u + _e992_);
                                let _e997_ = (_e996_ < 1536u);
                                if _e997_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e999_ = scan_scratch[_e996_];
                                let _e1000_ = (512u + _e992_);
                                let _e1001_ = (_e1000_ < 1536u);
                                if _e1001_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1003_ = scan_scratch[_e1000_];
                                let _e1004_ = (768u + _e992_);
                                let _e1005_ = (_e1004_ < 1536u);
                                if _e1005_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1007_ = scan_scratch[_e1004_];
                                let _e1008_ = (1024u + _e992_);
                                let _e1009_ = (_e1008_ < 1536u);
                                if _e1009_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1011_ = scan_scratch[_e1008_];
                                let _e1012_ = (1280u + _e992_);
                                let _e1013_ = (_e1012_ < 1536u);
                                if _e1013_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1015_ = scan_scratch[_e1012_];
                                let _e1017_ = select(255u, _e988_, (_e988_ < 255u));
                                let _e1018_ = (_e1017_ < 1536u);
                                if _e1018_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1020_ = scan_scratch[_e1017_];
                                let _e1021_ = (256u + _e1017_);
                                let _e1022_ = (_e1021_ < 1536u);
                                if _e1022_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1024_ = scan_scratch[_e1021_];
                                let _e1025_ = (512u + _e1017_);
                                let _e1026_ = (_e1025_ < 1536u);
                                if _e1026_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1028_ = scan_scratch[_e1025_];
                                let _e1029_ = (768u + _e1017_);
                                let _e1030_ = (_e1029_ < 1536u);
                                if _e1030_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1032_ = scan_scratch[_e1029_];
                                let _e1033_ = (1024u + _e1017_);
                                let _e1034_ = (_e1033_ < 1536u);
                                if _e1034_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1036_ = scan_scratch[_e1033_];
                                let _e1037_ = (1280u + _e1017_);
                                let _e1038_ = (_e1037_ < 1536u);
                                if _e1038_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                let _e1040_ = scan_scratch[_e1037_];
                                if _e993_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e992_] = _e1020_;
                                if _e997_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e996_] = _e1024_;
                                if _e1001_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1000_] = _e1028_;
                                if _e1005_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1004_] = _e1032_;
                                if _e1009_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1008_] = _e1036_;
                                if _e1013_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1012_] = _e1040_;
                                if _e1018_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1017_] = ((_e995_ * _e1020_) + (_e999_ * _e1028_));
                                if _e1022_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1021_] = ((_e995_ * _e1024_) + (_e999_ * _e1032_));
                                if _e1026_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1025_] = ((_e1003_ * _e1020_) + (_e1007_ * _e1028_));
                                if _e1030_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1029_] = ((_e1003_ * _e1024_) + (_e1007_ * _e1032_));
                                if _e1034_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1033_] = (((_e995_ * _e1036_) + (_e999_ * _e1040_)) + _e1011_);
                                if _e1038_ {
                                } else {
                                    phi_12077_ = true;
                                    break;
                                }
                                scan_scratch[_e1037_] = (((_e1003_ * _e1036_) + (_e1007_ * _e1040_)) + _e1015_);
                            }
                            workgroupBarrier();
                            phi_2564_ = (_e982_ >> bitcast<u32>(1i));
                        } else {
                            phi_2564_ = u32();
                        }
                        let _e1064_ = phi_2564_;
                        phi_2563_ = _e1064_;
                        phi_12077_ = _e973_;
                        if !(_e983_) {
                            break;
                        }
                    }
                    let _e1067_ = phi_12077_;
                    phi_12412_ = _e1067_;
                    if _e1067_ {
                        break;
                    }
                    if _e867_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e871_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e875_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e879_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e883_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e1068_ = scan_scratch[_e882_];
                    if _e887_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e1069_ = scan_scratch[_e886_];
                    workgroupBarrier();
                    if _e158_ {
                        let _e1070_ = (_e194_.x - _e1069_);
                        let _e1073_ = ((_e592_ * _e1068_) + (_e593_ * _e1070_));
                        let _e1077_ = ((_e1069_ + (_e593_ * _e1068_)) + (_e594_ * _e1070_));
                        let _e1088_ = bitcast<u32>(((((_e641_ * _e1077_) + (_e659_ * _e1073_)) + (_e677_ * ((_e194_.x - (_e578_ * _e1073_)) - _e1077_))) + (_e695_ * _e194_.x)));
                        let _e1089_ = (_e1088_ & 2139095040u);
                        if (_e1089_ == 2139095040u) {
                            phi_3106_ = true;
                        } else {
                            phi_3106_ = (_e1089_ == 0u);
                        }
                        let _e1093_ = phi_3106_;
                        let _e1096_ = bitcast<u32>(_e1073_);
                        let _e1097_ = (_e1096_ & 2139095040u);
                        if (_e1097_ == 2139095040u) {
                            phi_3122_ = true;
                        } else {
                            phi_3122_ = (_e1097_ == 0u);
                        }
                        let _e1101_ = phi_3122_;
                        phi_3130_ = type_20_(bitcast<f32>(select(_e1088_, 0u, _e1093_)), bitcast<f32>(select(_e1096_, 0u, _e1101_)));
                    } else {
                        phi_3130_ = type_20_(0f, 0f);
                    }
                    let _e1106_ = phi_3130_;
                    if _e158_ {
                        let _e1110_ = (_e593_ + _e593_);
                        let _e1111_ = (_e594_ + _e594_);
                        phi_3153_ = type_23_(((_e592_ + _e592_) - 1f), -(_e1110_), _e1110_, (1f - _e1111_), (_e1110_ * _e194_.y), (_e1111_ * _e194_.y));
                    } else {
                        phi_3153_ = type_23_(1f, 0f, 0f, 1f, 0f, 0f);
                    }
                    let _e1120_ = phi_3153_;
                    if _e867_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e1120_.member;
                    if _e871_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e1120_.member_1_;
                    if _e875_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e874_] = _e1120_.member_2_;
                    if _e879_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e878_] = _e1120_.member_3_;
                    if _e883_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e882_] = _e1120_.member_4_;
                    if _e887_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e886_] = _e1120_.member_5_;
                    workgroupBarrier();
                    phi_3213_ = 1u;
                    loop {
                        let _e1128_ = phi_3213_;
                        let _e1129_ = (_e1128_ < 256u);
                        if _e1129_ {
                            let _e1131_ = (_e1128_ << bitcast<u32>(1i));
                            let _e1134_ = (((_e113_ + 1u) * _e1131_) - 1u);
                            if (_e1134_ < 256u) {
                                let _e1136_ = (_e1134_ - _e1128_);
                                let _e1138_ = select(255u, _e1136_, (_e1136_ < 255u));
                                if (_e1138_ < 1536u) {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1141_ = scan_scratch[_e1138_];
                                let _e1142_ = (256u + _e1138_);
                                if (_e1142_ < 1536u) {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1145_ = scan_scratch[_e1142_];
                                let _e1146_ = (512u + _e1138_);
                                if (_e1146_ < 1536u) {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1149_ = scan_scratch[_e1146_];
                                let _e1150_ = (768u + _e1138_);
                                if (_e1150_ < 1536u) {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1153_ = scan_scratch[_e1150_];
                                let _e1154_ = (1024u + _e1138_);
                                if (_e1154_ < 1536u) {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1157_ = scan_scratch[_e1154_];
                                let _e1158_ = (1280u + _e1138_);
                                if (_e1158_ < 1536u) {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1161_ = scan_scratch[_e1158_];
                                let _e1163_ = select(255u, _e1134_, (_e1134_ < 255u));
                                let _e1164_ = (_e1163_ < 1536u);
                                if _e1164_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1166_ = scan_scratch[_e1163_];
                                let _e1167_ = (256u + _e1163_);
                                let _e1168_ = (_e1167_ < 1536u);
                                if _e1168_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1170_ = scan_scratch[_e1167_];
                                let _e1171_ = (512u + _e1163_);
                                let _e1172_ = (_e1171_ < 1536u);
                                if _e1172_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1174_ = scan_scratch[_e1171_];
                                let _e1175_ = (768u + _e1163_);
                                let _e1176_ = (_e1175_ < 1536u);
                                if _e1176_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1178_ = scan_scratch[_e1175_];
                                let _e1179_ = (1024u + _e1163_);
                                let _e1180_ = (_e1179_ < 1536u);
                                if _e1180_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1182_ = scan_scratch[_e1179_];
                                let _e1183_ = (1280u + _e1163_);
                                let _e1184_ = (_e1183_ < 1536u);
                                if _e1184_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                let _e1186_ = scan_scratch[_e1183_];
                                if _e1164_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                scan_scratch[_e1163_] = ((_e1166_ * _e1141_) + (_e1170_ * _e1149_));
                                if _e1168_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                scan_scratch[_e1167_] = ((_e1166_ * _e1145_) + (_e1170_ * _e1153_));
                                if _e1172_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                scan_scratch[_e1171_] = ((_e1174_ * _e1141_) + (_e1178_ * _e1149_));
                                if _e1176_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                scan_scratch[_e1175_] = ((_e1174_ * _e1145_) + (_e1178_ * _e1153_));
                                if _e1180_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                scan_scratch[_e1179_] = (((_e1166_ * _e1157_) + (_e1170_ * _e1161_)) + _e1182_);
                                if _e1184_ {
                                } else {
                                    phi_12092_ = true;
                                    break;
                                }
                                scan_scratch[_e1183_] = (((_e1174_ * _e1157_) + (_e1178_ * _e1161_)) + _e1186_);
                            }
                            workgroupBarrier();
                            phi_3214_ = _e1131_;
                        } else {
                            phi_3214_ = u32();
                        }
                        let _e1208_ = phi_3214_;
                        phi_3213_ = _e1208_;
                        phi_12092_ = _e1067_;
                        if !(_e1129_) {
                            break;
                        }
                    }
                    let _e1211_ = phi_12092_;
                    phi_12412_ = _e1211_;
                    if _e1211_ {
                        break;
                    }
                    if _e974_ {
                        scan_scratch[255] = 1f;
                        scan_scratch[511] = 0f;
                        scan_scratch[767] = 0f;
                        scan_scratch[1023] = 1f;
                        scan_scratch[1279] = 0f;
                        scan_scratch[1535] = 0f;
                    }
                    workgroupBarrier();
                    phi_3476_ = 128u;
                    loop {
                        let _e1219_ = phi_3476_;
                        let _e1220_ = (_e1219_ > 0u);
                        if _e1220_ {
                            let _e1225_ = (((_e113_ + 1u) * (_e1219_ << bitcast<u32>(1i))) - 1u);
                            if (_e1225_ < 256u) {
                                let _e1227_ = (_e1225_ - _e1219_);
                                let _e1229_ = select(255u, _e1227_, (_e1227_ < 255u));
                                let _e1230_ = (_e1229_ < 1536u);
                                if _e1230_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1232_ = scan_scratch[_e1229_];
                                let _e1233_ = (256u + _e1229_);
                                let _e1234_ = (_e1233_ < 1536u);
                                if _e1234_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1236_ = scan_scratch[_e1233_];
                                let _e1237_ = (512u + _e1229_);
                                let _e1238_ = (_e1237_ < 1536u);
                                if _e1238_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1240_ = scan_scratch[_e1237_];
                                let _e1241_ = (768u + _e1229_);
                                let _e1242_ = (_e1241_ < 1536u);
                                if _e1242_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1244_ = scan_scratch[_e1241_];
                                let _e1245_ = (1024u + _e1229_);
                                let _e1246_ = (_e1245_ < 1536u);
                                if _e1246_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1248_ = scan_scratch[_e1245_];
                                let _e1249_ = (1280u + _e1229_);
                                let _e1250_ = (_e1249_ < 1536u);
                                if _e1250_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1252_ = scan_scratch[_e1249_];
                                let _e1254_ = select(255u, _e1225_, (_e1225_ < 255u));
                                let _e1255_ = (_e1254_ < 1536u);
                                if _e1255_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1257_ = scan_scratch[_e1254_];
                                let _e1258_ = (256u + _e1254_);
                                let _e1259_ = (_e1258_ < 1536u);
                                if _e1259_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1261_ = scan_scratch[_e1258_];
                                let _e1262_ = (512u + _e1254_);
                                let _e1263_ = (_e1262_ < 1536u);
                                if _e1263_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1265_ = scan_scratch[_e1262_];
                                let _e1266_ = (768u + _e1254_);
                                let _e1267_ = (_e1266_ < 1536u);
                                if _e1267_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1269_ = scan_scratch[_e1266_];
                                let _e1270_ = (1024u + _e1254_);
                                let _e1271_ = (_e1270_ < 1536u);
                                if _e1271_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1273_ = scan_scratch[_e1270_];
                                let _e1274_ = (1280u + _e1254_);
                                let _e1275_ = (_e1274_ < 1536u);
                                if _e1275_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                let _e1277_ = scan_scratch[_e1274_];
                                if _e1230_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1229_] = _e1257_;
                                if _e1234_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1233_] = _e1261_;
                                if _e1238_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1237_] = _e1265_;
                                if _e1242_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1241_] = _e1269_;
                                if _e1246_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1245_] = _e1273_;
                                if _e1250_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1249_] = _e1277_;
                                if _e1255_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1254_] = ((_e1232_ * _e1257_) + (_e1236_ * _e1265_));
                                if _e1259_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1258_] = ((_e1232_ * _e1261_) + (_e1236_ * _e1269_));
                                if _e1263_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1262_] = ((_e1240_ * _e1257_) + (_e1244_ * _e1265_));
                                if _e1267_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1266_] = ((_e1240_ * _e1261_) + (_e1244_ * _e1269_));
                                if _e1271_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1270_] = (((_e1232_ * _e1273_) + (_e1236_ * _e1277_)) + _e1248_);
                                if _e1275_ {
                                } else {
                                    phi_12112_ = true;
                                    break;
                                }
                                scan_scratch[_e1274_] = (((_e1240_ * _e1273_) + (_e1244_ * _e1277_)) + _e1252_);
                            }
                            workgroupBarrier();
                            phi_3477_ = (_e1219_ >> bitcast<u32>(1i));
                        } else {
                            phi_3477_ = u32();
                        }
                        let _e1301_ = phi_3477_;
                        phi_3476_ = _e1301_;
                        phi_12112_ = _e1211_;
                        if !(_e1220_) {
                            break;
                        }
                    }
                    let _e1304_ = phi_12112_;
                    phi_12412_ = _e1304_;
                    if _e1304_ {
                        break;
                    }
                    if _e867_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e871_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e875_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e879_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    if _e883_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e1305_ = scan_scratch[_e882_];
                    if _e887_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    let _e1306_ = scan_scratch[_e886_];
                    workgroupBarrier();
                    if _e158_ {
                        let _e1307_ = (_e194_.y - _e1306_);
                        let _e1310_ = ((_e592_ * _e1305_) + (_e593_ * _e1307_));
                        let _e1314_ = ((_e1306_ + (_e593_ * _e1305_)) + (_e594_ * _e1307_));
                        let _e1325_ = bitcast<u32>(((((_e641_ * _e1314_) + (_e659_ * _e1310_)) + (_e677_ * ((_e194_.y - (_e578_ * _e1310_)) - _e1314_))) + (_e695_ * _e194_.y)));
                        let _e1326_ = (_e1325_ & 2139095040u);
                        if (_e1326_ == 2139095040u) {
                            phi_4019_ = true;
                        } else {
                            phi_4019_ = (_e1326_ == 0u);
                        }
                        let _e1330_ = phi_4019_;
                        let _e1333_ = bitcast<u32>(_e1310_);
                        let _e1334_ = (_e1333_ & 2139095040u);
                        if (_e1334_ == 2139095040u) {
                            phi_4035_ = true;
                        } else {
                            phi_4035_ = (_e1334_ == 0u);
                        }
                        let _e1338_ = phi_4035_;
                        phi_4043_ = type_20_(bitcast<f32>(select(_e1325_, 0u, _e1330_)), bitcast<f32>(select(_e1333_, 0u, _e1338_)));
                    } else {
                        phi_4043_ = type_20_(0f, 0f);
                    }
                    let _e1343_ = phi_4043_;
                    if _e867_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e1106_.member;
                    if _e871_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e1343_.member;
                    if _e875_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e874_] = _e1106_.member_1_;
                    if _e879_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e878_] = _e1343_.member_1_;
                    if _e883_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e882_] = 0f;
                    if _e887_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e886_] = 0f;
                    workgroupBarrier();
                    if (_e113_ == 0u) {
                        phi_4108_ = type_15_(0u, select(_e154_, 256u, (256u < _e154_)));
                        phi_4111_ = type_22_(array<f32, 4>(0f, 0f, 0f, 0f));
                        phi_4113_ = type_22_(array<f32, 4>(0f, 0f, 0f, 0f));
                        loop {
                            let _e1349_ = phi_4108_;
                            let _e1351_ = phi_4111_;
                            let _e1353_ = phi_4113_;
                            if (_e1349_.member < _e1349_.member_1_) {
                                phi_4109_ = type_15_((_e1349_.member + 1u), _e1349_.member_1_);
                                phi_4136_ = type_15_(1u, _e1349_.member);
                            } else {
                                phi_4109_ = _e1349_;
                                phi_4136_ = type_15_(0u, u32());
                            }
                            let _e1366_ = phi_4109_;
                            let _e1368_ = phi_4136_;
                            let _e1372_ = (bitcast<i32>(_e1368_.member) != 0i);
                            if _e1372_ {
                                if (_e1368_.member_1_ < 1536u) {
                                } else {
                                    phi_12127_ = true;
                                    break;
                                }
                                let _e1375_ = scan_scratch[_e1368_.member_1_];
                                let _e1376_ = (256u + _e1368_.member_1_);
                                if (_e1376_ < 1536u) {
                                } else {
                                    phi_12127_ = true;
                                    break;
                                }
                                let _e1379_ = scan_scratch[_e1376_];
                                let _e1380_ = (512u + _e1368_.member_1_);
                                if (_e1380_ < 1536u) {
                                } else {
                                    phi_12127_ = true;
                                    break;
                                }
                                let _e1383_ = scan_scratch[_e1380_];
                                let _e1384_ = (768u + _e1368_.member_1_);
                                if (_e1384_ < 1536u) {
                                } else {
                                    phi_12127_ = true;
                                    break;
                                }
                                let _e1387_ = scan_scratch[_e1384_];
                                let _e1388_ = (_e734_ * _e1375_);
                                let _e1389_ = (_e734_ * _e1379_);
                                let _e1392_ = bitcast<u32>((_e623_ * _e1383_));
                                let _e1393_ = (_e1392_ & 2139095040u);
                                let _e1394_ = (_e1393_ == 2139095040u);
                                if _e1394_ {
                                    phi_4191_ = 0u;
                                } else {
                                    if (_e1393_ == 0u) {
                                        phi_4190_ = (_e1392_ & 2147483648u);
                                    } else {
                                        phi_4190_ = _e1392_;
                                    }
                                    let _e1398_ = phi_4190_;
                                    phi_4191_ = _e1398_;
                                }
                                let _e1400_ = phi_4191_;
                                if _e1394_ {
                                    phi_4205_ = 0u;
                                } else {
                                    if (_e1393_ == 0u) {
                                        phi_4204_ = (_e1392_ & 2147483648u);
                                    } else {
                                        phi_4204_ = _e1392_;
                                    }
                                    let _e1405_ = phi_4204_;
                                    phi_4205_ = _e1405_;
                                }
                                let _e1407_ = phi_4205_;
                                if _e1394_ {
                                    phi_4219_ = 0u;
                                } else {
                                    if (_e1393_ == 0u) {
                                        phi_4218_ = (_e1392_ & 2147483648u);
                                    } else {
                                        phi_4218_ = _e1392_;
                                    }
                                    let _e1412_ = phi_4218_;
                                    phi_4219_ = _e1412_;
                                }
                                let _e1414_ = phi_4219_;
                                if _e1394_ {
                                    phi_4233_ = 0u;
                                } else {
                                    if (_e1393_ == 0u) {
                                        phi_4232_ = (_e1392_ & 2147483648u);
                                    } else {
                                        phi_4232_ = _e1392_;
                                    }
                                    let _e1419_ = phi_4232_;
                                    phi_4233_ = _e1419_;
                                }
                                let _e1421_ = phi_4233_;
                                let _e1439_ = (_e556_ <= 0f);
                                let _e1444_ = (1f - select(select(select(0f, 1f, (1f <= _e460_)), _e460_, (_e460_ <= 1f)), 0f, (_e460_ <= 0f)));
                                let _e1445_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1400_)) & 2147483647u))), 0f, _e1439_));
                                let _e1446_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1407_)) & 2147483647u))), 0f, _e1439_));
                                let _e1447_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1414_)) & 2147483647u))), 0f, _e1439_));
                                let _e1448_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e1421_)) & 2147483647u))), 0f, _e1439_));
                                if (_e1445_ != _e1445_) {
                                    phi_8396_ = true;
                                } else {
                                    phi_8396_ = (0f >= _e1445_);
                                }
                                let _e1452_ = phi_8396_;
                                let _e1453_ = select(_e1445_, 0f, _e1452_);
                                if (_e1446_ != _e1446_) {
                                    phi_8411_ = true;
                                } else {
                                    phi_8411_ = (0f >= _e1446_);
                                }
                                let _e1457_ = phi_8411_;
                                let _e1458_ = select(_e1446_, 0f, _e1457_);
                                if (_e1447_ != _e1447_) {
                                    phi_8426_ = true;
                                } else {
                                    phi_8426_ = (0f >= _e1447_);
                                }
                                let _e1462_ = phi_8426_;
                                let _e1463_ = select(_e1447_, 0f, _e1462_);
                                if (_e1448_ != _e1448_) {
                                    phi_8441_ = true;
                                } else {
                                    phi_8441_ = (0f >= _e1448_);
                                }
                                let _e1467_ = phi_8441_;
                                let _e1468_ = select(_e1448_, 0f, _e1467_);
                                if (_e1453_ != _e1453_) {
                                    phi_8456_ = true;
                                } else {
                                    phi_8456_ = (1f <= _e1453_);
                                }
                                let _e1472_ = phi_8456_;
                                let _e1473_ = select(_e1453_, 1f, _e1472_);
                                if (_e1458_ != _e1458_) {
                                    phi_8471_ = true;
                                } else {
                                    phi_8471_ = (1f <= _e1458_);
                                }
                                let _e1477_ = phi_8471_;
                                let _e1478_ = select(_e1458_, 1f, _e1477_);
                                if (_e1463_ != _e1463_) {
                                    phi_8486_ = true;
                                } else {
                                    phi_8486_ = (1f <= _e1463_);
                                }
                                let _e1482_ = phi_8486_;
                                let _e1483_ = select(_e1463_, 1f, _e1482_);
                                if (_e1468_ != _e1468_) {
                                    phi_8501_ = true;
                                } else {
                                    phi_8501_ = (1f <= _e1468_);
                                }
                                let _e1487_ = phi_8501_;
                                let _e1488_ = select(_e1468_, 1f, _e1487_);
                                let _e1491_ = (_e538_ * _e1353_.member[0]);
                                let _e1494_ = (_e538_ * _e1353_.member[1]);
                                let _e1497_ = (_e538_ * _e1353_.member[2]);
                                let _e1500_ = (_e538_ * _e1353_.member[3]);
                                let _e1501_ = (1f - _e562_);
                                let _e1502_ = (select(select(select(0f, 1f, (1f <= _e478_)), _e478_, (_e478_ <= 1f)), 0f, (_e478_ <= 0f)) * _e1501_);
                                let _e1504_ = (_e1388_ - (_e1502_ * _e1491_));
                                let _e1506_ = (_e1388_ - (_e1502_ * _e1494_));
                                let _e1508_ = (_e1388_ - (_e1502_ * _e1497_));
                                let _e1510_ = (_e1388_ - (_e1502_ * _e1500_));
                                if (_e1504_ != _e1504_) {
                                    phi_8516_ = true;
                                } else {
                                    phi_8516_ = (-1f >= _e1504_);
                                }
                                let _e1514_ = phi_8516_;
                                let _e1515_ = select(_e1504_, -1f, _e1514_);
                                if (_e1506_ != _e1506_) {
                                    phi_8531_ = true;
                                } else {
                                    phi_8531_ = (-1f >= _e1506_);
                                }
                                let _e1519_ = phi_8531_;
                                let _e1520_ = select(_e1506_, -1f, _e1519_);
                                if (_e1508_ != _e1508_) {
                                    phi_8546_ = true;
                                } else {
                                    phi_8546_ = (-1f >= _e1508_);
                                }
                                let _e1524_ = phi_8546_;
                                let _e1525_ = select(_e1508_, -1f, _e1524_);
                                if (_e1510_ != _e1510_) {
                                    phi_8561_ = true;
                                } else {
                                    phi_8561_ = (-1f >= _e1510_);
                                }
                                let _e1529_ = phi_8561_;
                                let _e1530_ = select(_e1510_, -1f, _e1529_);
                                if (_e1515_ != _e1515_) {
                                    phi_8576_ = true;
                                } else {
                                    phi_8576_ = (1f <= _e1515_);
                                }
                                let _e1534_ = phi_8576_;
                                let _e1535_ = select(_e1515_, 1f, _e1534_);
                                if (_e1520_ != _e1520_) {
                                    phi_8591_ = true;
                                } else {
                                    phi_8591_ = (1f <= _e1520_);
                                }
                                let _e1539_ = phi_8591_;
                                let _e1540_ = select(_e1520_, 1f, _e1539_);
                                if (_e1525_ != _e1525_) {
                                    phi_8606_ = true;
                                } else {
                                    phi_8606_ = (1f <= _e1525_);
                                }
                                let _e1544_ = phi_8606_;
                                let _e1545_ = select(_e1525_, 1f, _e1544_);
                                if (_e1530_ != _e1530_) {
                                    phi_8621_ = true;
                                } else {
                                    phi_8621_ = (1f <= _e1530_);
                                }
                                let _e1549_ = phi_8621_;
                                let _e1550_ = select(_e1530_, 1f, _e1549_);
                                let _e1569_ = bitcast<f32>((bitcast<u32>(_e1504_) & 2147483647u));
                                let _e1572_ = bitcast<f32>((bitcast<u32>(_e1506_) & 2147483647u));
                                let _e1575_ = bitcast<f32>((bitcast<u32>(_e1508_) & 2147483647u));
                                let _e1578_ = bitcast<f32>((bitcast<u32>(_e1510_) & 2147483647u));
                                if (_e1569_ != _e1569_) {
                                    phi_8636_ = true;
                                } else {
                                    phi_8636_ = (1000000000000000f <= _e1569_);
                                }
                                let _e1582_ = phi_8636_;
                                let _e1583_ = select(_e1569_, 1000000000000000f, _e1582_);
                                if (_e1572_ != _e1572_) {
                                    phi_8651_ = true;
                                } else {
                                    phi_8651_ = (1000000000000000f <= _e1572_);
                                }
                                let _e1587_ = phi_8651_;
                                let _e1588_ = select(_e1572_, 1000000000000000f, _e1587_);
                                if (_e1575_ != _e1575_) {
                                    phi_8666_ = true;
                                } else {
                                    phi_8666_ = (1000000000000000f <= _e1575_);
                                }
                                let _e1592_ = phi_8666_;
                                let _e1593_ = select(_e1575_, 1000000000000000f, _e1592_);
                                if (_e1578_ != _e1578_) {
                                    phi_8681_ = true;
                                } else {
                                    phi_8681_ = (1000000000000000f <= _e1578_);
                                }
                                let _e1597_ = phi_8681_;
                                let _e1598_ = select(_e1578_, 1000000000000000f, _e1597_);
                                let _e1599_ = (_e1583_ * _e1583_);
                                let _e1600_ = (_e1588_ * _e1588_);
                                let _e1601_ = (_e1593_ * _e1593_);
                                let _e1602_ = (_e1598_ * _e1598_);
                                let _e1603_ = (_e1599_ + 0.0000010000001f);
                                let _e1604_ = (_e1600_ + 0.0000010000001f);
                                let _e1605_ = (_e1601_ + 0.0000010000001f);
                                let _e1606_ = (_e1602_ + 0.0000010000001f);
                                if (_e1603_ != _e1603_) {
                                    phi_8696_ = true;
                                } else {
                                    phi_8696_ = (0.000000000001f >= _e1603_);
                                }
                                let _e1610_ = phi_8696_;
                                let _e1611_ = select(_e1603_, 0.000000000001f, _e1610_);
                                if (_e1604_ != _e1604_) {
                                    phi_8711_ = true;
                                } else {
                                    phi_8711_ = (0.000000000001f >= _e1604_);
                                }
                                let _e1615_ = phi_8711_;
                                let _e1616_ = select(_e1604_, 0.000000000001f, _e1615_);
                                if (_e1605_ != _e1605_) {
                                    phi_8726_ = true;
                                } else {
                                    phi_8726_ = (0.000000000001f >= _e1605_);
                                }
                                let _e1620_ = phi_8726_;
                                let _e1621_ = select(_e1605_, 0.000000000001f, _e1620_);
                                if (_e1606_ != _e1606_) {
                                    phi_8741_ = true;
                                } else {
                                    phi_8741_ = (0.000000000001f >= _e1606_);
                                }
                                let _e1625_ = phi_8741_;
                                let _e1626_ = select(_e1606_, 0.000000000001f, _e1625_);
                                let _e1627_ = (_e1611_ * 0.5f);
                                let _e1628_ = (_e1616_ * 0.5f);
                                let _e1629_ = (_e1621_ * 0.5f);
                                let _e1630_ = (_e1626_ * 0.5f);
                                let _e1635_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1611_) >> bitcast<u32>(1i))));
                                let _e1640_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1616_) >> bitcast<u32>(1i))));
                                let _e1645_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1621_) >> bitcast<u32>(1i))));
                                let _e1650_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e1626_) >> bitcast<u32>(1i))));
                                let _e1663_ = (_e1635_ * (1.5f - (_e1627_ * (_e1635_ * _e1635_))));
                                let _e1664_ = (_e1640_ * (1.5f - (_e1628_ * (_e1640_ * _e1640_))));
                                let _e1665_ = (_e1645_ * (1.5f - (_e1629_ * (_e1645_ * _e1645_))));
                                let _e1666_ = (_e1650_ * (1.5f - (_e1630_ * (_e1650_ * _e1650_))));
                                let _e1679_ = (_e1663_ * (1.5f - (_e1627_ * (_e1663_ * _e1663_))));
                                let _e1680_ = (_e1664_ * (1.5f - (_e1628_ * (_e1664_ * _e1664_))));
                                let _e1681_ = (_e1665_ * (1.5f - (_e1629_ * (_e1665_ * _e1665_))));
                                let _e1682_ = (_e1666_ * (1.5f - (_e1630_ * (_e1666_ * _e1666_))));
                                let _e1699_ = (_e1603_ * (_e1679_ * (1.5f - (_e1627_ * (_e1679_ * _e1679_)))));
                                let _e1700_ = (_e1604_ * (_e1680_ * (1.5f - (_e1628_ * (_e1680_ * _e1680_)))));
                                let _e1701_ = (_e1605_ * (_e1681_ * (1.5f - (_e1629_ * (_e1681_ * _e1681_)))));
                                let _e1702_ = (_e1606_ * (_e1682_ * (1.5f - (_e1630_ * (_e1682_ * _e1682_)))));
                                let _e1703_ = (_e1699_ + 0.001f);
                                let _e1704_ = (_e1700_ + 0.001f);
                                let _e1705_ = (_e1701_ + 0.001f);
                                let _e1706_ = (_e1702_ + 0.001f);
                                if (_e1703_ != _e1703_) {
                                    phi_8756_ = true;
                                } else {
                                    phi_8756_ = (0.000001f >= _e1703_);
                                }
                                let _e1710_ = phi_8756_;
                                let _e1711_ = select(_e1703_, 0.000001f, _e1710_);
                                if (_e1704_ != _e1704_) {
                                    phi_8771_ = true;
                                } else {
                                    phi_8771_ = (0.000001f >= _e1704_);
                                }
                                let _e1715_ = phi_8771_;
                                let _e1716_ = select(_e1704_, 0.000001f, _e1715_);
                                if (_e1705_ != _e1705_) {
                                    phi_8786_ = true;
                                } else {
                                    phi_8786_ = (0.000001f >= _e1705_);
                                }
                                let _e1720_ = phi_8786_;
                                let _e1721_ = select(_e1705_, 0.000001f, _e1720_);
                                if (_e1706_ != _e1706_) {
                                    phi_8801_ = true;
                                } else {
                                    phi_8801_ = (0.000001f >= _e1706_);
                                }
                                let _e1725_ = phi_8801_;
                                let _e1726_ = select(_e1706_, 0.000001f, _e1725_);
                                let _e1729_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1711_)));
                                let _e1732_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1716_)));
                                let _e1735_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1721_)));
                                let _e1738_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1726_)));
                                let _e1748_ = (_e1729_ + (_e1729_ * (1f - (_e1711_ * _e1729_))));
                                let _e1750_ = (_e1732_ + (_e1732_ * (1f - (_e1716_ * _e1732_))));
                                let _e1752_ = (_e1735_ + (_e1735_ * (1f - (_e1721_ * _e1735_))));
                                let _e1754_ = (_e1738_ + (_e1738_ * (1f - (_e1726_ * _e1738_))));
                                let _e1764_ = (_e1748_ + (_e1748_ * (1f - (_e1711_ * _e1748_))));
                                let _e1766_ = (_e1750_ + (_e1750_ * (1f - (_e1716_ * _e1750_))));
                                let _e1768_ = (_e1752_ + (_e1752_ * (1f - (_e1721_ * _e1752_))));
                                let _e1770_ = (_e1754_ + (_e1754_ * (1f - (_e1726_ * _e1754_))));
                                let _e1780_ = (_e1764_ + (_e1764_ * (1f - (_e1711_ * _e1764_))));
                                let _e1782_ = (_e1766_ + (_e1766_ * (1f - (_e1716_ * _e1766_))));
                                let _e1784_ = (_e1768_ + (_e1768_ * (1f - (_e1721_ * _e1768_))));
                                let _e1786_ = (_e1770_ + (_e1770_ * (1f - (_e1726_ * _e1770_))));
                                let _e1787_ = (_e1699_ + _e1583_);
                                let _e1788_ = (_e1700_ + _e1588_);
                                let _e1789_ = (_e1701_ + _e1593_);
                                let _e1790_ = (_e1702_ + _e1598_);
                                if (_e1787_ != _e1787_) {
                                    phi_8816_ = true;
                                } else {
                                    phi_8816_ = (0.000001f >= _e1787_);
                                }
                                let _e1794_ = phi_8816_;
                                let _e1795_ = select(_e1787_, 0.000001f, _e1794_);
                                if (_e1788_ != _e1788_) {
                                    phi_8831_ = true;
                                } else {
                                    phi_8831_ = (0.000001f >= _e1788_);
                                }
                                let _e1799_ = phi_8831_;
                                let _e1800_ = select(_e1788_, 0.000001f, _e1799_);
                                if (_e1789_ != _e1789_) {
                                    phi_8846_ = true;
                                } else {
                                    phi_8846_ = (0.000001f >= _e1789_);
                                }
                                let _e1804_ = phi_8846_;
                                let _e1805_ = select(_e1789_, 0.000001f, _e1804_);
                                if (_e1790_ != _e1790_) {
                                    phi_8861_ = true;
                                } else {
                                    phi_8861_ = (0.000001f >= _e1790_);
                                }
                                let _e1809_ = phi_8861_;
                                let _e1810_ = select(_e1790_, 0.000001f, _e1809_);
                                let _e1813_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1795_)));
                                let _e1816_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1800_)));
                                let _e1819_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1805_)));
                                let _e1822_ = bitcast<f32>((2129859011u - bitcast<u32>(_e1810_)));
                                let _e1832_ = (_e1813_ + (_e1813_ * (1f - (_e1795_ * _e1813_))));
                                let _e1834_ = (_e1816_ + (_e1816_ * (1f - (_e1800_ * _e1816_))));
                                let _e1836_ = (_e1819_ + (_e1819_ * (1f - (_e1805_ * _e1819_))));
                                let _e1838_ = (_e1822_ + (_e1822_ * (1f - (_e1810_ * _e1822_))));
                                let _e1848_ = (_e1832_ + (_e1832_ * (1f - (_e1795_ * _e1832_))));
                                let _e1850_ = (_e1834_ + (_e1834_ * (1f - (_e1800_ * _e1834_))));
                                let _e1852_ = (_e1836_ + (_e1836_ * (1f - (_e1805_ * _e1836_))));
                                let _e1854_ = (_e1838_ + (_e1838_ * (1f - (_e1810_ * _e1838_))));
                                let _e1891_ = (_e1569_ <= 1000000000000000f);
                                let _e1893_ = (_e1572_ <= 1000000000000000f);
                                let _e1895_ = (_e1575_ <= 1000000000000000f);
                                let _e1897_ = (_e1578_ <= 1000000000000000f);
                                let _e1928_ = select(select(select(_e1569_, (0.5f * (_e1583_ + (_e1599_ * _e1780_))), _e1891_), select(-0.0005f, -(((0.0005f * (_e1583_ * (_e1848_ + (_e1848_ * (1f - (_e1795_ * _e1848_)))))) * (1f + (_e1583_ * _e1780_)))), _e1891_), (_e1504_ <= 0f)), 0f, (_e1569_ <= 0f));
                                let _e1930_ = select(select(select(_e1572_, (0.5f * (_e1588_ + (_e1600_ * _e1782_))), _e1893_), select(-0.0005f, -(((0.0005f * (_e1588_ * (_e1850_ + (_e1850_ * (1f - (_e1800_ * _e1850_)))))) * (1f + (_e1588_ * _e1782_)))), _e1893_), (_e1506_ <= 0f)), 0f, (_e1572_ <= 0f));
                                let _e1932_ = select(select(select(_e1575_, (0.5f * (_e1593_ + (_e1601_ * _e1784_))), _e1895_), select(-0.0005f, -(((0.0005f * (_e1593_ * (_e1852_ + (_e1852_ * (1f - (_e1805_ * _e1852_)))))) * (1f + (_e1593_ * _e1784_)))), _e1895_), (_e1508_ <= 0f)), 0f, (_e1575_ <= 0f));
                                let _e1934_ = select(select(select(_e1578_, (0.5f * (_e1598_ + (_e1602_ * _e1786_))), _e1897_), select(-0.0005f, -(((0.0005f * (_e1598_ * (_e1854_ + (_e1854_ * (1f - (_e1810_ * _e1854_)))))) * (1f + (_e1598_ * _e1786_)))), _e1897_), (_e1510_ <= 0f)), 0f, (_e1578_ <= 0f));
                                let _e1947_ = (_e562_ <= 0f);
                                let _e1972_ = bitcast<u32>((((1f - (_e562_ * _e1473_)) * _e1491_) + (_e1473_ * select(((_e1501_ * _e1928_) + (_e562_ * (_e1535_ * (1f - (0.33333334f * (_e1535_ * _e1535_)))))), _e1928_, _e1947_))));
                                let _e1973_ = (_e1972_ & 2139095040u);
                                if (_e1973_ == 2139095040u) {
                                    phi_4939_ = 0u;
                                } else {
                                    if (_e1973_ == 0u) {
                                        phi_4938_ = (_e1972_ & 2147483648u);
                                    } else {
                                        phi_4938_ = _e1972_;
                                    }
                                    let _e1978_ = phi_4938_;
                                    phi_4939_ = _e1978_;
                                }
                                let _e1980_ = phi_4939_;
                                let _e1981_ = bitcast<f32>(_e1980_);
                                let _e1982_ = bitcast<u32>((((1f - (_e562_ * _e1478_)) * _e1494_) + (_e1478_ * select(((_e1501_ * _e1930_) + (_e562_ * (_e1540_ * (1f - (0.33333334f * (_e1540_ * _e1540_)))))), _e1930_, _e1947_))));
                                let _e1983_ = (_e1982_ & 2139095040u);
                                if (_e1983_ == 2139095040u) {
                                    phi_4953_ = 0u;
                                } else {
                                    if (_e1983_ == 0u) {
                                        phi_4952_ = (_e1982_ & 2147483648u);
                                    } else {
                                        phi_4952_ = _e1982_;
                                    }
                                    let _e1988_ = phi_4952_;
                                    phi_4953_ = _e1988_;
                                }
                                let _e1990_ = phi_4953_;
                                let _e1992_ = bitcast<u32>((((1f - (_e562_ * _e1483_)) * _e1497_) + (_e1483_ * select(((_e1501_ * _e1932_) + (_e562_ * (_e1545_ * (1f - (0.33333334f * (_e1545_ * _e1545_)))))), _e1932_, _e1947_))));
                                let _e1993_ = (_e1992_ & 2139095040u);
                                if (_e1993_ == 2139095040u) {
                                    phi_4967_ = 0u;
                                } else {
                                    if (_e1993_ == 0u) {
                                        phi_4966_ = (_e1992_ & 2147483648u);
                                    } else {
                                        phi_4966_ = _e1992_;
                                    }
                                    let _e1998_ = phi_4966_;
                                    phi_4967_ = _e1998_;
                                }
                                let _e2000_ = phi_4967_;
                                let _e2002_ = bitcast<u32>((((1f - (_e562_ * _e1488_)) * _e1500_) + (_e1488_ * select(((_e1501_ * _e1934_) + (_e562_ * (_e1550_ * (1f - (0.33333334f * (_e1550_ * _e1550_)))))), _e1934_, _e1947_))));
                                let _e2003_ = (_e2002_ & 2139095040u);
                                if (_e2003_ == 2139095040u) {
                                    phi_4981_ = 0u;
                                } else {
                                    if (_e2003_ == 0u) {
                                        phi_4980_ = (_e2002_ & 2147483648u);
                                    } else {
                                        phi_4980_ = _e2002_;
                                    }
                                    let _e2008_ = phi_4980_;
                                    phi_4981_ = _e2008_;
                                }
                                let _e2010_ = phi_4981_;
                                let _e2014_ = bitcast<u32>((_e623_ * _e1387_));
                                let _e2015_ = (_e2014_ & 2139095040u);
                                let _e2016_ = (_e2015_ == 2139095040u);
                                if _e2016_ {
                                    phi_5001_ = 0u;
                                } else {
                                    if (_e2015_ == 0u) {
                                        phi_5000_ = (_e2014_ & 2147483648u);
                                    } else {
                                        phi_5000_ = _e2014_;
                                    }
                                    let _e2020_ = phi_5000_;
                                    phi_5001_ = _e2020_;
                                }
                                let _e2022_ = phi_5001_;
                                if _e2016_ {
                                    phi_5015_ = 0u;
                                } else {
                                    if (_e2015_ == 0u) {
                                        phi_5014_ = (_e2014_ & 2147483648u);
                                    } else {
                                        phi_5014_ = _e2014_;
                                    }
                                    let _e2027_ = phi_5014_;
                                    phi_5015_ = _e2027_;
                                }
                                let _e2029_ = phi_5015_;
                                if _e2016_ {
                                    phi_5029_ = 0u;
                                } else {
                                    if (_e2015_ == 0u) {
                                        phi_5028_ = (_e2014_ & 2147483648u);
                                    } else {
                                        phi_5028_ = _e2014_;
                                    }
                                    let _e2034_ = phi_5028_;
                                    phi_5029_ = _e2034_;
                                }
                                let _e2036_ = phi_5029_;
                                if _e2016_ {
                                    phi_5043_ = 0u;
                                } else {
                                    if (_e2015_ == 0u) {
                                        phi_5042_ = (_e2014_ & 2147483648u);
                                    } else {
                                        phi_5042_ = _e2014_;
                                    }
                                    let _e2041_ = phi_5042_;
                                    phi_5043_ = _e2041_;
                                }
                                let _e2043_ = phi_5043_;
                                let _e2065_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2022_)) & 2147483647u))), 0f, _e1439_));
                                let _e2066_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2029_)) & 2147483647u))), 0f, _e1439_));
                                let _e2067_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2036_)) & 2147483647u))), 0f, _e1439_));
                                let _e2068_ = (_e1444_ + select((_e556_ * bitcast<f32>((bitcast<u32>(bitcast<f32>(_e2043_)) & 2147483647u))), 0f, _e1439_));
                                if (_e2065_ != _e2065_) {
                                    phi_8996_ = true;
                                } else {
                                    phi_8996_ = (0f >= _e2065_);
                                }
                                let _e2072_ = phi_8996_;
                                let _e2073_ = select(_e2065_, 0f, _e2072_);
                                if (_e2066_ != _e2066_) {
                                    phi_9011_ = true;
                                } else {
                                    phi_9011_ = (0f >= _e2066_);
                                }
                                let _e2077_ = phi_9011_;
                                let _e2078_ = select(_e2066_, 0f, _e2077_);
                                if (_e2067_ != _e2067_) {
                                    phi_9026_ = true;
                                } else {
                                    phi_9026_ = (0f >= _e2067_);
                                }
                                let _e2082_ = phi_9026_;
                                let _e2083_ = select(_e2067_, 0f, _e2082_);
                                if (_e2068_ != _e2068_) {
                                    phi_9041_ = true;
                                } else {
                                    phi_9041_ = (0f >= _e2068_);
                                }
                                let _e2087_ = phi_9041_;
                                let _e2088_ = select(_e2068_, 0f, _e2087_);
                                if (_e2073_ != _e2073_) {
                                    phi_9056_ = true;
                                } else {
                                    phi_9056_ = (1f <= _e2073_);
                                }
                                let _e2092_ = phi_9056_;
                                let _e2093_ = select(_e2073_, 1f, _e2092_);
                                if (_e2078_ != _e2078_) {
                                    phi_9071_ = true;
                                } else {
                                    phi_9071_ = (1f <= _e2078_);
                                }
                                let _e2097_ = phi_9071_;
                                let _e2098_ = select(_e2078_, 1f, _e2097_);
                                if (_e2083_ != _e2083_) {
                                    phi_9086_ = true;
                                } else {
                                    phi_9086_ = (1f <= _e2083_);
                                }
                                let _e2102_ = phi_9086_;
                                let _e2103_ = select(_e2083_, 1f, _e2102_);
                                if (_e2088_ != _e2088_) {
                                    phi_9101_ = true;
                                } else {
                                    phi_9101_ = (1f <= _e2088_);
                                }
                                let _e2107_ = phi_9101_;
                                let _e2108_ = select(_e2088_, 1f, _e2107_);
                                let _e2111_ = (_e538_ * _e1351_.member[0]);
                                let _e2114_ = (_e538_ * _e1351_.member[1]);
                                let _e2117_ = (_e538_ * _e1351_.member[2]);
                                let _e2120_ = (_e538_ * _e1351_.member[3]);
                                let _e2122_ = (_e1389_ - (_e1502_ * _e2111_));
                                let _e2124_ = (_e1389_ - (_e1502_ * _e2114_));
                                let _e2126_ = (_e1389_ - (_e1502_ * _e2117_));
                                let _e2128_ = (_e1389_ - (_e1502_ * _e2120_));
                                if (_e2122_ != _e2122_) {
                                    phi_9116_ = true;
                                } else {
                                    phi_9116_ = (-1f >= _e2122_);
                                }
                                let _e2132_ = phi_9116_;
                                let _e2133_ = select(_e2122_, -1f, _e2132_);
                                if (_e2124_ != _e2124_) {
                                    phi_9131_ = true;
                                } else {
                                    phi_9131_ = (-1f >= _e2124_);
                                }
                                let _e2137_ = phi_9131_;
                                let _e2138_ = select(_e2124_, -1f, _e2137_);
                                if (_e2126_ != _e2126_) {
                                    phi_9146_ = true;
                                } else {
                                    phi_9146_ = (-1f >= _e2126_);
                                }
                                let _e2142_ = phi_9146_;
                                let _e2143_ = select(_e2126_, -1f, _e2142_);
                                if (_e2128_ != _e2128_) {
                                    phi_9161_ = true;
                                } else {
                                    phi_9161_ = (-1f >= _e2128_);
                                }
                                let _e2147_ = phi_9161_;
                                let _e2148_ = select(_e2128_, -1f, _e2147_);
                                if (_e2133_ != _e2133_) {
                                    phi_9176_ = true;
                                } else {
                                    phi_9176_ = (1f <= _e2133_);
                                }
                                let _e2152_ = phi_9176_;
                                let _e2153_ = select(_e2133_, 1f, _e2152_);
                                if (_e2138_ != _e2138_) {
                                    phi_9191_ = true;
                                } else {
                                    phi_9191_ = (1f <= _e2138_);
                                }
                                let _e2157_ = phi_9191_;
                                let _e2158_ = select(_e2138_, 1f, _e2157_);
                                if (_e2143_ != _e2143_) {
                                    phi_9206_ = true;
                                } else {
                                    phi_9206_ = (1f <= _e2143_);
                                }
                                let _e2162_ = phi_9206_;
                                let _e2163_ = select(_e2143_, 1f, _e2162_);
                                if (_e2148_ != _e2148_) {
                                    phi_9221_ = true;
                                } else {
                                    phi_9221_ = (1f <= _e2148_);
                                }
                                let _e2167_ = phi_9221_;
                                let _e2168_ = select(_e2148_, 1f, _e2167_);
                                let _e2187_ = bitcast<f32>((bitcast<u32>(_e2122_) & 2147483647u));
                                let _e2190_ = bitcast<f32>((bitcast<u32>(_e2124_) & 2147483647u));
                                let _e2193_ = bitcast<f32>((bitcast<u32>(_e2126_) & 2147483647u));
                                let _e2196_ = bitcast<f32>((bitcast<u32>(_e2128_) & 2147483647u));
                                if (_e2187_ != _e2187_) {
                                    phi_9236_ = true;
                                } else {
                                    phi_9236_ = (1000000000000000f <= _e2187_);
                                }
                                let _e2200_ = phi_9236_;
                                let _e2201_ = select(_e2187_, 1000000000000000f, _e2200_);
                                if (_e2190_ != _e2190_) {
                                    phi_9251_ = true;
                                } else {
                                    phi_9251_ = (1000000000000000f <= _e2190_);
                                }
                                let _e2205_ = phi_9251_;
                                let _e2206_ = select(_e2190_, 1000000000000000f, _e2205_);
                                if (_e2193_ != _e2193_) {
                                    phi_9266_ = true;
                                } else {
                                    phi_9266_ = (1000000000000000f <= _e2193_);
                                }
                                let _e2210_ = phi_9266_;
                                let _e2211_ = select(_e2193_, 1000000000000000f, _e2210_);
                                if (_e2196_ != _e2196_) {
                                    phi_9281_ = true;
                                } else {
                                    phi_9281_ = (1000000000000000f <= _e2196_);
                                }
                                let _e2215_ = phi_9281_;
                                let _e2216_ = select(_e2196_, 1000000000000000f, _e2215_);
                                let _e2217_ = (_e2201_ * _e2201_);
                                let _e2218_ = (_e2206_ * _e2206_);
                                let _e2219_ = (_e2211_ * _e2211_);
                                let _e2220_ = (_e2216_ * _e2216_);
                                let _e2221_ = (_e2217_ + 0.0000010000001f);
                                let _e2222_ = (_e2218_ + 0.0000010000001f);
                                let _e2223_ = (_e2219_ + 0.0000010000001f);
                                let _e2224_ = (_e2220_ + 0.0000010000001f);
                                if (_e2221_ != _e2221_) {
                                    phi_9296_ = true;
                                } else {
                                    phi_9296_ = (0.000000000001f >= _e2221_);
                                }
                                let _e2228_ = phi_9296_;
                                let _e2229_ = select(_e2221_, 0.000000000001f, _e2228_);
                                if (_e2222_ != _e2222_) {
                                    phi_9311_ = true;
                                } else {
                                    phi_9311_ = (0.000000000001f >= _e2222_);
                                }
                                let _e2233_ = phi_9311_;
                                let _e2234_ = select(_e2222_, 0.000000000001f, _e2233_);
                                if (_e2223_ != _e2223_) {
                                    phi_9326_ = true;
                                } else {
                                    phi_9326_ = (0.000000000001f >= _e2223_);
                                }
                                let _e2238_ = phi_9326_;
                                let _e2239_ = select(_e2223_, 0.000000000001f, _e2238_);
                                if (_e2224_ != _e2224_) {
                                    phi_9341_ = true;
                                } else {
                                    phi_9341_ = (0.000000000001f >= _e2224_);
                                }
                                let _e2243_ = phi_9341_;
                                let _e2244_ = select(_e2224_, 0.000000000001f, _e2243_);
                                let _e2245_ = (_e2229_ * 0.5f);
                                let _e2246_ = (_e2234_ * 0.5f);
                                let _e2247_ = (_e2239_ * 0.5f);
                                let _e2248_ = (_e2244_ * 0.5f);
                                let _e2253_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2229_) >> bitcast<u32>(1i))));
                                let _e2258_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2234_) >> bitcast<u32>(1i))));
                                let _e2263_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2239_) >> bitcast<u32>(1i))));
                                let _e2268_ = bitcast<f32>((1597463007u - (bitcast<u32>(_e2244_) >> bitcast<u32>(1i))));
                                let _e2281_ = (_e2253_ * (1.5f - (_e2245_ * (_e2253_ * _e2253_))));
                                let _e2282_ = (_e2258_ * (1.5f - (_e2246_ * (_e2258_ * _e2258_))));
                                let _e2283_ = (_e2263_ * (1.5f - (_e2247_ * (_e2263_ * _e2263_))));
                                let _e2284_ = (_e2268_ * (1.5f - (_e2248_ * (_e2268_ * _e2268_))));
                                let _e2297_ = (_e2281_ * (1.5f - (_e2245_ * (_e2281_ * _e2281_))));
                                let _e2298_ = (_e2282_ * (1.5f - (_e2246_ * (_e2282_ * _e2282_))));
                                let _e2299_ = (_e2283_ * (1.5f - (_e2247_ * (_e2283_ * _e2283_))));
                                let _e2300_ = (_e2284_ * (1.5f - (_e2248_ * (_e2284_ * _e2284_))));
                                let _e2317_ = (_e2221_ * (_e2297_ * (1.5f - (_e2245_ * (_e2297_ * _e2297_)))));
                                let _e2318_ = (_e2222_ * (_e2298_ * (1.5f - (_e2246_ * (_e2298_ * _e2298_)))));
                                let _e2319_ = (_e2223_ * (_e2299_ * (1.5f - (_e2247_ * (_e2299_ * _e2299_)))));
                                let _e2320_ = (_e2224_ * (_e2300_ * (1.5f - (_e2248_ * (_e2300_ * _e2300_)))));
                                let _e2321_ = (_e2317_ + 0.001f);
                                let _e2322_ = (_e2318_ + 0.001f);
                                let _e2323_ = (_e2319_ + 0.001f);
                                let _e2324_ = (_e2320_ + 0.001f);
                                if (_e2321_ != _e2321_) {
                                    phi_9356_ = true;
                                } else {
                                    phi_9356_ = (0.000001f >= _e2321_);
                                }
                                let _e2328_ = phi_9356_;
                                let _e2329_ = select(_e2321_, 0.000001f, _e2328_);
                                if (_e2322_ != _e2322_) {
                                    phi_9371_ = true;
                                } else {
                                    phi_9371_ = (0.000001f >= _e2322_);
                                }
                                let _e2333_ = phi_9371_;
                                let _e2334_ = select(_e2322_, 0.000001f, _e2333_);
                                if (_e2323_ != _e2323_) {
                                    phi_9386_ = true;
                                } else {
                                    phi_9386_ = (0.000001f >= _e2323_);
                                }
                                let _e2338_ = phi_9386_;
                                let _e2339_ = select(_e2323_, 0.000001f, _e2338_);
                                if (_e2324_ != _e2324_) {
                                    phi_9401_ = true;
                                } else {
                                    phi_9401_ = (0.000001f >= _e2324_);
                                }
                                let _e2343_ = phi_9401_;
                                let _e2344_ = select(_e2324_, 0.000001f, _e2343_);
                                let _e2347_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2329_)));
                                let _e2350_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2334_)));
                                let _e2353_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2339_)));
                                let _e2356_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2344_)));
                                let _e2366_ = (_e2347_ + (_e2347_ * (1f - (_e2329_ * _e2347_))));
                                let _e2368_ = (_e2350_ + (_e2350_ * (1f - (_e2334_ * _e2350_))));
                                let _e2370_ = (_e2353_ + (_e2353_ * (1f - (_e2339_ * _e2353_))));
                                let _e2372_ = (_e2356_ + (_e2356_ * (1f - (_e2344_ * _e2356_))));
                                let _e2382_ = (_e2366_ + (_e2366_ * (1f - (_e2329_ * _e2366_))));
                                let _e2384_ = (_e2368_ + (_e2368_ * (1f - (_e2334_ * _e2368_))));
                                let _e2386_ = (_e2370_ + (_e2370_ * (1f - (_e2339_ * _e2370_))));
                                let _e2388_ = (_e2372_ + (_e2372_ * (1f - (_e2344_ * _e2372_))));
                                let _e2398_ = (_e2382_ + (_e2382_ * (1f - (_e2329_ * _e2382_))));
                                let _e2400_ = (_e2384_ + (_e2384_ * (1f - (_e2334_ * _e2384_))));
                                let _e2402_ = (_e2386_ + (_e2386_ * (1f - (_e2339_ * _e2386_))));
                                let _e2404_ = (_e2388_ + (_e2388_ * (1f - (_e2344_ * _e2388_))));
                                let _e2405_ = (_e2317_ + _e2201_);
                                let _e2406_ = (_e2318_ + _e2206_);
                                let _e2407_ = (_e2319_ + _e2211_);
                                let _e2408_ = (_e2320_ + _e2216_);
                                if (_e2405_ != _e2405_) {
                                    phi_9416_ = true;
                                } else {
                                    phi_9416_ = (0.000001f >= _e2405_);
                                }
                                let _e2412_ = phi_9416_;
                                let _e2413_ = select(_e2405_, 0.000001f, _e2412_);
                                if (_e2406_ != _e2406_) {
                                    phi_9431_ = true;
                                } else {
                                    phi_9431_ = (0.000001f >= _e2406_);
                                }
                                let _e2417_ = phi_9431_;
                                let _e2418_ = select(_e2406_, 0.000001f, _e2417_);
                                if (_e2407_ != _e2407_) {
                                    phi_9446_ = true;
                                } else {
                                    phi_9446_ = (0.000001f >= _e2407_);
                                }
                                let _e2422_ = phi_9446_;
                                let _e2423_ = select(_e2407_, 0.000001f, _e2422_);
                                if (_e2408_ != _e2408_) {
                                    phi_9461_ = true;
                                } else {
                                    phi_9461_ = (0.000001f >= _e2408_);
                                }
                                let _e2427_ = phi_9461_;
                                let _e2428_ = select(_e2408_, 0.000001f, _e2427_);
                                let _e2431_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2413_)));
                                let _e2434_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2418_)));
                                let _e2437_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2423_)));
                                let _e2440_ = bitcast<f32>((2129859011u - bitcast<u32>(_e2428_)));
                                let _e2450_ = (_e2431_ + (_e2431_ * (1f - (_e2413_ * _e2431_))));
                                let _e2452_ = (_e2434_ + (_e2434_ * (1f - (_e2418_ * _e2434_))));
                                let _e2454_ = (_e2437_ + (_e2437_ * (1f - (_e2423_ * _e2437_))));
                                let _e2456_ = (_e2440_ + (_e2440_ * (1f - (_e2428_ * _e2440_))));
                                let _e2466_ = (_e2450_ + (_e2450_ * (1f - (_e2413_ * _e2450_))));
                                let _e2468_ = (_e2452_ + (_e2452_ * (1f - (_e2418_ * _e2452_))));
                                let _e2470_ = (_e2454_ + (_e2454_ * (1f - (_e2423_ * _e2454_))));
                                let _e2472_ = (_e2456_ + (_e2456_ * (1f - (_e2428_ * _e2456_))));
                                let _e2509_ = (_e2187_ <= 1000000000000000f);
                                let _e2511_ = (_e2190_ <= 1000000000000000f);
                                let _e2513_ = (_e2193_ <= 1000000000000000f);
                                let _e2515_ = (_e2196_ <= 1000000000000000f);
                                let _e2546_ = select(select(select(_e2187_, (0.5f * (_e2201_ + (_e2217_ * _e2398_))), _e2509_), select(-0.0005f, -(((0.0005f * (_e2201_ * (_e2466_ + (_e2466_ * (1f - (_e2413_ * _e2466_)))))) * (1f + (_e2201_ * _e2398_)))), _e2509_), (_e2122_ <= 0f)), 0f, (_e2187_ <= 0f));
                                let _e2548_ = select(select(select(_e2190_, (0.5f * (_e2206_ + (_e2218_ * _e2400_))), _e2511_), select(-0.0005f, -(((0.0005f * (_e2206_ * (_e2468_ + (_e2468_ * (1f - (_e2418_ * _e2468_)))))) * (1f + (_e2206_ * _e2400_)))), _e2511_), (_e2124_ <= 0f)), 0f, (_e2190_ <= 0f));
                                let _e2550_ = select(select(select(_e2193_, (0.5f * (_e2211_ + (_e2219_ * _e2402_))), _e2513_), select(-0.0005f, -(((0.0005f * (_e2211_ * (_e2470_ + (_e2470_ * (1f - (_e2423_ * _e2470_)))))) * (1f + (_e2211_ * _e2402_)))), _e2513_), (_e2126_ <= 0f)), 0f, (_e2193_ <= 0f));
                                let _e2552_ = select(select(select(_e2196_, (0.5f * (_e2216_ + (_e2220_ * _e2404_))), _e2515_), select(-0.0005f, -(((0.0005f * (_e2216_ * (_e2472_ + (_e2472_ * (1f - (_e2428_ * _e2472_)))))) * (1f + (_e2216_ * _e2404_)))), _e2515_), (_e2128_ <= 0f)), 0f, (_e2196_ <= 0f));
                                let _e2589_ = bitcast<u32>((((1f - (_e562_ * _e2093_)) * _e2111_) + (_e2093_ * select(((_e1501_ * _e2546_) + (_e562_ * (_e2153_ * (1f - (0.33333334f * (_e2153_ * _e2153_)))))), _e2546_, _e1947_))));
                                let _e2590_ = (_e2589_ & 2139095040u);
                                if (_e2590_ == 2139095040u) {
                                    phi_5749_ = 0u;
                                } else {
                                    if (_e2590_ == 0u) {
                                        phi_5748_ = (_e2589_ & 2147483648u);
                                    } else {
                                        phi_5748_ = _e2589_;
                                    }
                                    let _e2595_ = phi_5748_;
                                    phi_5749_ = _e2595_;
                                }
                                let _e2597_ = phi_5749_;
                                let _e2598_ = bitcast<f32>(_e2597_);
                                let _e2599_ = bitcast<u32>((((1f - (_e562_ * _e2098_)) * _e2114_) + (_e2098_ * select(((_e1501_ * _e2548_) + (_e562_ * (_e2158_ * (1f - (0.33333334f * (_e2158_ * _e2158_)))))), _e2548_, _e1947_))));
                                let _e2600_ = (_e2599_ & 2139095040u);
                                if (_e2600_ == 2139095040u) {
                                    phi_5763_ = 0u;
                                } else {
                                    if (_e2600_ == 0u) {
                                        phi_5762_ = (_e2599_ & 2147483648u);
                                    } else {
                                        phi_5762_ = _e2599_;
                                    }
                                    let _e2605_ = phi_5762_;
                                    phi_5763_ = _e2605_;
                                }
                                let _e2607_ = phi_5763_;
                                let _e2609_ = bitcast<u32>((((1f - (_e562_ * _e2103_)) * _e2117_) + (_e2103_ * select(((_e1501_ * _e2550_) + (_e562_ * (_e2163_ * (1f - (0.33333334f * (_e2163_ * _e2163_)))))), _e2550_, _e1947_))));
                                let _e2610_ = (_e2609_ & 2139095040u);
                                if (_e2610_ == 2139095040u) {
                                    phi_5777_ = 0u;
                                } else {
                                    if (_e2610_ == 0u) {
                                        phi_5776_ = (_e2609_ & 2147483648u);
                                    } else {
                                        phi_5776_ = _e2609_;
                                    }
                                    let _e2615_ = phi_5776_;
                                    phi_5777_ = _e2615_;
                                }
                                let _e2617_ = phi_5777_;
                                let _e2619_ = bitcast<u32>((((1f - (_e562_ * _e2108_)) * _e2120_) + (_e2108_ * select(((_e1501_ * _e2552_) + (_e562_ * (_e2168_ * (1f - (0.33333334f * (_e2168_ * _e2168_)))))), _e2552_, _e1947_))));
                                let _e2620_ = (_e2619_ & 2139095040u);
                                if (_e2620_ == 2139095040u) {
                                    phi_5791_ = 0u;
                                } else {
                                    if (_e2620_ == 0u) {
                                        phi_5790_ = (_e2619_ & 2147483648u);
                                    } else {
                                        phi_5790_ = _e2619_;
                                    }
                                    let _e2625_ = phi_5790_;
                                    phi_5791_ = _e2625_;
                                }
                                let _e2627_ = phi_5791_;
                                let _e2632_ = (0.5f * (_e1981_ + _e2598_));
                                let _e2639_ = (1024u + _e1368_.member_1_);
                                if (_e2639_ < 1536u) {
                                } else {
                                    phi_12127_ = true;
                                    break;
                                }
                                scan_scratch[_e2639_] = (_e1981_ + (_e568_ * (_e2632_ - _e1981_)));
                                let _e2642_ = (1280u + _e1368_.member_1_);
                                if (_e2642_ < 1536u) {
                                } else {
                                    phi_12127_ = true;
                                    break;
                                }
                                scan_scratch[_e2642_] = (_e2598_ + (_e568_ * (_e2632_ - _e2598_)));
                                phi_4112_ = type_22_(array<f32, 4>(_e2598_, bitcast<f32>(_e2607_), bitcast<f32>(_e2617_), bitcast<f32>(_e2627_)));
                                phi_4114_ = type_22_(array<f32, 4>(_e1981_, bitcast<f32>(_e1990_), bitcast<f32>(_e2000_), bitcast<f32>(_e2010_)));
                            } else {
                                phi_4112_ = type_22_();
                                phi_4114_ = type_22_();
                            }
                            let _e2646_ = phi_4112_;
                            let _e2648_ = phi_4114_;
                            phi_4108_ = _e1366_;
                            phi_4111_ = _e2646_;
                            phi_4113_ = _e2648_;
                            phi_12127_ = _e1304_;
                            if !(_e1372_) {
                                break;
                            }
                        }
                        let _e2651_ = phi_12127_;
                        phi_12412_ = _e2651_;
                        if _e2651_ {
                            break;
                        }
                        phi_12809_ = _e2651_;
                    } else {
                        phi_12809_ = _e1304_;
                    }
                    let _e2653_ = phi_12809_;
                    workgroupBarrier();
                    if _e158_ {
                        if _e883_ {
                        } else {
                            phi_12412_ = true;
                            break;
                        }
                        let _e2654_ = scan_scratch[_e882_];
                        phi_5875_ = _e2654_;
                    } else {
                        phi_5875_ = 0f;
                    }
                    let _e2656_ = phi_5875_;
                    if _e158_ {
                        if _e887_ {
                        } else {
                            phi_12412_ = true;
                            break;
                        }
                        let _e2657_ = scan_scratch[_e886_];
                        phi_5887_ = _e2657_;
                    } else {
                        phi_5887_ = 0f;
                    }
                    let _e2659_ = phi_5887_;
                    let _e2661_ = (1f - (_e773_ * _e2656_));
                    if (_e2661_ != _e2661_) {
                        phi_9596_ = true;
                    } else {
                        phi_9596_ = (0.05f >= _e2661_);
                    }
                    let _e2665_ = phi_9596_;
                    let _e2668_ = (1f - (_e773_ * _e2659_));
                    if (_e2668_ != _e2668_) {
                        phi_9611_ = true;
                    } else {
                        phi_9611_ = (0.05f >= _e2668_);
                    }
                    let _e2672_ = phi_9611_;
                    if _e158_ {
                        let _e2678_ = bitcast<u32>((((_e1106_.member * _e812_) * select(_e2661_, 0.05f, _e2665_)) + (_e851_ * _e2656_)));
                        let _e2679_ = (_e2678_ & 2139095040u);
                        if (_e2679_ == 2139095040u) {
                            phi_5911_ = true;
                        } else {
                            phi_5911_ = (_e2679_ == 0u);
                        }
                        let _e2683_ = phi_5911_;
                        phi_5917_ = bitcast<f32>(select(_e2678_, 0u, _e2683_));
                    } else {
                        phi_5917_ = 0f;
                    }
                    let _e2687_ = phi_5917_;
                    if _e158_ {
                        let _e2692_ = bitcast<u32>((((_e1343_.member * _e812_) * select(_e2668_, 0.05f, _e2672_)) + (_e851_ * _e2659_)));
                        let _e2693_ = (_e2692_ & 2139095040u);
                        if (_e2693_ == 2139095040u) {
                            phi_5935_ = true;
                        } else {
                            phi_5935_ = (_e2693_ == 0u);
                        }
                        let _e2697_ = phi_5935_;
                        phi_5941_ = bitcast<f32>(select(_e2692_, 0u, _e2697_));
                    } else {
                        phi_5941_ = 0f;
                    }
                    let _e2701_ = phi_5941_;
                    workgroupBarrier();
                    if _e867_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e865_] = _e2687_;
                    if _e871_ {
                    } else {
                        phi_12412_ = true;
                        break;
                    }
                    scan_scratch[_e870_] = _e2701_;
                    workgroupBarrier();
                    let _e2702_ = (_e113_ > 0u);
                    if _e2702_ {
                        if _e158_ {
                            let _e2703_ = (_e113_ - 1u);
                            if (_e2703_ < 1536u) {
                            } else {
                                phi_12412_ = true;
                                break;
                            }
                            let _e2706_ = scan_scratch[_e2703_];
                            phi_5970_ = _e2706_;
                        } else {
                            phi_5970_ = f32();
                        }
                        let _e2708_ = phi_5970_;
                        phi_5972_ = _e2708_;
                        phi_5973_ = select(true, false, _e158_);
                    } else {
                        phi_5972_ = f32();
                        phi_5973_ = true;
                    }
                    let _e2711_ = phi_5972_;
                    let _e2713_ = phi_5973_;
                    let _e2714_ = select(_e2711_, 0f, _e2713_);
                    if _e2702_ {
                        if _e158_ {
                            let _e2715_ = (_e113_ + 255u);
                            if (_e2715_ < 1536u) {
                            } else {
                                phi_12412_ = true;
                                break;
                            }
                            let _e2718_ = scan_scratch[_e2715_];
                            phi_5994_ = _e2718_;
                        } else {
                            phi_5994_ = f32();
                        }
                        let _e2720_ = phi_5994_;
                        phi_5996_ = _e2720_;
                        phi_5997_ = select(true, false, _e158_);
                    } else {
                        phi_5996_ = f32();
                        phi_5997_ = true;
                    }
                    let _e2723_ = phi_5996_;
                    let _e2725_ = phi_5997_;
                    let _e2726_ = select(_e2723_, 0f, _e2725_);
                    let _e2731_ = select(0f, _e2714_, (bitcast<f32>((bitcast<u32>(_e2714_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2732_ = (_e2731_ != _e2731_);
                    if _e2732_ {
                        phi_9626_ = true;
                    } else {
                        phi_9626_ = (-1000000000000000000f >= _e2731_);
                    }
                    let _e2735_ = phi_9626_;
                    let _e2736_ = select(_e2731_, -1000000000000000000f, _e2735_);
                    if _e2732_ {
                        phi_9641_ = true;
                    } else {
                        phi_9641_ = (-1000000000000000000f >= _e2731_);
                    }
                    let _e2739_ = phi_9641_;
                    let _e2740_ = select(_e2731_, -1000000000000000000f, _e2739_);
                    if _e2732_ {
                        phi_9656_ = true;
                    } else {
                        phi_9656_ = (-1000000000000000000f >= _e2731_);
                    }
                    let _e2743_ = phi_9656_;
                    let _e2744_ = select(_e2731_, -1000000000000000000f, _e2743_);
                    if _e2732_ {
                        phi_9671_ = true;
                    } else {
                        phi_9671_ = (-1000000000000000000f >= _e2731_);
                    }
                    let _e2747_ = phi_9671_;
                    let _e2748_ = select(_e2731_, -1000000000000000000f, _e2747_);
                    if (_e2736_ != _e2736_) {
                        phi_9686_ = true;
                    } else {
                        phi_9686_ = (1000000000000000000f <= _e2736_);
                    }
                    let _e2752_ = phi_9686_;
                    let _e2753_ = select(_e2736_, 1000000000000000000f, _e2752_);
                    if (_e2740_ != _e2740_) {
                        phi_9701_ = true;
                    } else {
                        phi_9701_ = (1000000000000000000f <= _e2740_);
                    }
                    let _e2757_ = phi_9701_;
                    let _e2758_ = select(_e2740_, 1000000000000000000f, _e2757_);
                    if (_e2744_ != _e2744_) {
                        phi_9716_ = true;
                    } else {
                        phi_9716_ = (1000000000000000000f <= _e2744_);
                    }
                    let _e2762_ = phi_9716_;
                    let _e2763_ = select(_e2744_, 1000000000000000000f, _e2762_);
                    if (_e2748_ != _e2748_) {
                        phi_9731_ = true;
                    } else {
                        phi_9731_ = (1000000000000000000f <= _e2748_);
                    }
                    let _e2767_ = phi_9731_;
                    let _e2768_ = select(_e2748_, 1000000000000000000f, _e2767_);
                    let _e2773_ = select(0f, _e2687_, (bitcast<f32>((bitcast<u32>(_e2687_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e2774_ = (_e2773_ != _e2773_);
                    if _e2774_ {
                        phi_9746_ = true;
                    } else {
                        phi_9746_ = (-1000000000000000000f >= _e2773_);
                    }
                    let _e2777_ = phi_9746_;
                    let _e2778_ = select(_e2773_, -1000000000000000000f, _e2777_);
                    if _e2774_ {
                        phi_9761_ = true;
                    } else {
                        phi_9761_ = (-1000000000000000000f >= _e2773_);
                    }
                    let _e2781_ = phi_9761_;
                    let _e2782_ = select(_e2773_, -1000000000000000000f, _e2781_);
                    if _e2774_ {
                        phi_9776_ = true;
                    } else {
                        phi_9776_ = (-1000000000000000000f >= _e2773_);
                    }
                    let _e2785_ = phi_9776_;
                    let _e2786_ = select(_e2773_, -1000000000000000000f, _e2785_);
                    if _e2774_ {
                        phi_9791_ = true;
                    } else {
                        phi_9791_ = (-1000000000000000000f >= _e2773_);
                    }
                    let _e2789_ = phi_9791_;
                    let _e2790_ = select(_e2773_, -1000000000000000000f, _e2789_);
                    if (_e2778_ != _e2778_) {
                        phi_9806_ = true;
                    } else {
                        phi_9806_ = (1000000000000000000f <= _e2778_);
                    }
                    let _e2794_ = phi_9806_;
                    let _e2795_ = select(_e2778_, 1000000000000000000f, _e2794_);
                    if (_e2782_ != _e2782_) {
                        phi_9821_ = true;
                    } else {
                        phi_9821_ = (1000000000000000000f <= _e2782_);
                    }
                    let _e2799_ = phi_9821_;
                    let _e2800_ = select(_e2782_, 1000000000000000000f, _e2799_);
                    if (_e2786_ != _e2786_) {
                        phi_9836_ = true;
                    } else {
                        phi_9836_ = (1000000000000000000f <= _e2786_);
                    }
                    let _e2804_ = phi_9836_;
                    let _e2805_ = select(_e2786_, 1000000000000000000f, _e2804_);
                    if (_e2790_ != _e2790_) {
                        phi_9851_ = true;
                    } else {
                        phi_9851_ = (1000000000000000000f <= _e2790_);
                    }
                    let _e2809_ = phi_9851_;
                    let _e2810_ = select(_e2790_, 1000000000000000000f, _e2809_);
                    let _e2811_ = array<f32, 4>(_e2795_, _e2800_, _e2805_, _e2810_);
                    let _e2812_ = (_e2753_ != _e2753_);
                    if _e2812_ {
                        phi_9866_ = true;
                    } else {
                        phi_9866_ = (_e2795_ <= _e2753_);
                    }
                    let _e2815_ = phi_9866_;
                    let _e2816_ = select(_e2753_, _e2795_, _e2815_);
                    let _e2817_ = (_e2758_ != _e2758_);
                    if _e2817_ {
                        phi_9881_ = true;
                    } else {
                        phi_9881_ = (_e2800_ <= _e2758_);
                    }
                    let _e2820_ = phi_9881_;
                    let _e2821_ = select(_e2758_, _e2800_, _e2820_);
                    let _e2822_ = (_e2763_ != _e2763_);
                    if _e2822_ {
                        phi_9896_ = true;
                    } else {
                        phi_9896_ = (_e2805_ <= _e2763_);
                    }
                    let _e2825_ = phi_9896_;
                    let _e2826_ = select(_e2763_, _e2805_, _e2825_);
                    let _e2827_ = (_e2768_ != _e2768_);
                    if _e2827_ {
                        phi_9911_ = true;
                    } else {
                        phi_9911_ = (_e2810_ <= _e2768_);
                    }
                    let _e2830_ = phi_9911_;
                    let _e2831_ = select(_e2768_, _e2810_, _e2830_);
                    if _e2812_ {
                        phi_9926_ = true;
                    } else {
                        phi_9926_ = (_e2795_ >= _e2753_);
                    }
                    let _e2834_ = phi_9926_;
                    let _e2835_ = select(_e2753_, _e2795_, _e2834_);
                    if _e2817_ {
                        phi_9941_ = true;
                    } else {
                        phi_9941_ = (_e2800_ >= _e2758_);
                    }
                    let _e2838_ = phi_9941_;
                    let _e2839_ = select(_e2758_, _e2800_, _e2838_);
                    if _e2822_ {
                        phi_9956_ = true;
                    } else {
                        phi_9956_ = (_e2805_ >= _e2763_);
                    }
                    let _e2842_ = phi_9956_;
                    let _e2843_ = select(_e2763_, _e2805_, _e2842_);
                    if _e2827_ {
                        phi_9971_ = true;
                    } else {
                        phi_9971_ = (_e2810_ >= _e2768_);
                    }
                    let _e2846_ = phi_9971_;
                    let _e2847_ = select(_e2768_, _e2810_, _e2846_);
                    let _e2848_ = (_e2816_ != _e2816_);
                    if _e2848_ {
                        phi_9986_ = true;
                    } else {
                        phi_9986_ = (-1f >= _e2816_);
                    }
                    let _e2851_ = phi_9986_;
                    let _e2852_ = select(_e2816_, -1f, _e2851_);
                    let _e2853_ = (_e2821_ != _e2821_);
                    if _e2853_ {
                        phi_10001_ = true;
                    } else {
                        phi_10001_ = (-1f >= _e2821_);
                    }
                    let _e2856_ = phi_10001_;
                    let _e2857_ = select(_e2821_, -1f, _e2856_);
                    let _e2858_ = (_e2826_ != _e2826_);
                    if _e2858_ {
                        phi_10016_ = true;
                    } else {
                        phi_10016_ = (-1f >= _e2826_);
                    }
                    let _e2861_ = phi_10016_;
                    let _e2862_ = select(_e2826_, -1f, _e2861_);
                    let _e2863_ = (_e2831_ != _e2831_);
                    if _e2863_ {
                        phi_10031_ = true;
                    } else {
                        phi_10031_ = (-1f >= _e2831_);
                    }
                    let _e2866_ = phi_10031_;
                    let _e2867_ = select(_e2831_, -1f, _e2866_);
                    if (_e2852_ != _e2852_) {
                        phi_10046_ = true;
                    } else {
                        phi_10046_ = (1f <= _e2852_);
                    }
                    let _e2871_ = phi_10046_;
                    let _e2872_ = select(_e2852_, 1f, _e2871_);
                    if (_e2857_ != _e2857_) {
                        phi_10061_ = true;
                    } else {
                        phi_10061_ = (1f <= _e2857_);
                    }
                    let _e2876_ = phi_10061_;
                    let _e2877_ = select(_e2857_, 1f, _e2876_);
                    if (_e2862_ != _e2862_) {
                        phi_10076_ = true;
                    } else {
                        phi_10076_ = (1f <= _e2862_);
                    }
                    let _e2881_ = phi_10076_;
                    let _e2882_ = select(_e2862_, 1f, _e2881_);
                    if (_e2867_ != _e2867_) {
                        phi_10091_ = true;
                    } else {
                        phi_10091_ = (1f <= _e2867_);
                    }
                    let _e2886_ = phi_10091_;
                    let _e2887_ = select(_e2867_, 1f, _e2886_);
                    let _e2888_ = (_e2835_ != _e2835_);
                    if _e2888_ {
                        phi_10106_ = true;
                    } else {
                        phi_10106_ = (-1f >= _e2835_);
                    }
                    let _e2891_ = phi_10106_;
                    let _e2892_ = select(_e2835_, -1f, _e2891_);
                    let _e2893_ = (_e2839_ != _e2839_);
                    if _e2893_ {
                        phi_10121_ = true;
                    } else {
                        phi_10121_ = (-1f >= _e2839_);
                    }
                    let _e2896_ = phi_10121_;
                    let _e2897_ = select(_e2839_, -1f, _e2896_);
                    let _e2898_ = (_e2843_ != _e2843_);
                    if _e2898_ {
                        phi_10136_ = true;
                    } else {
                        phi_10136_ = (-1f >= _e2843_);
                    }
                    let _e2901_ = phi_10136_;
                    let _e2902_ = select(_e2843_, -1f, _e2901_);
                    let _e2903_ = (_e2847_ != _e2847_);
                    if _e2903_ {
                        phi_10151_ = true;
                    } else {
                        phi_10151_ = (-1f >= _e2847_);
                    }
                    let _e2906_ = phi_10151_;
                    let _e2907_ = select(_e2847_, -1f, _e2906_);
                    if (_e2892_ != _e2892_) {
                        phi_10166_ = true;
                    } else {
                        phi_10166_ = (1f <= _e2892_);
                    }
                    let _e2911_ = phi_10166_;
                    let _e2912_ = select(_e2892_, 1f, _e2911_);
                    if (_e2897_ != _e2897_) {
                        phi_10181_ = true;
                    } else {
                        phi_10181_ = (1f <= _e2897_);
                    }
                    let _e2916_ = phi_10181_;
                    let _e2917_ = select(_e2897_, 1f, _e2916_);
                    if (_e2902_ != _e2902_) {
                        phi_10196_ = true;
                    } else {
                        phi_10196_ = (1f <= _e2902_);
                    }
                    let _e2921_ = phi_10196_;
                    let _e2922_ = select(_e2902_, 1f, _e2921_);
                    if (_e2907_ != _e2907_) {
                        phi_10211_ = true;
                    } else {
                        phi_10211_ = (1f <= _e2907_);
                    }
                    let _e2926_ = phi_10211_;
                    let _e2927_ = select(_e2907_, 1f, _e2926_);
                    let _e2932_ = (0.5f * (_e2912_ + _e2872_));
                    let _e2933_ = (0.5f * (_e2917_ + _e2877_));
                    let _e2934_ = (0.5f * (_e2922_ + _e2882_));
                    let _e2935_ = (0.5f * (_e2927_ + _e2887_));
                    let _e2936_ = (_e2912_ - _e2872_);
                    let _e2937_ = (_e2917_ - _e2877_);
                    let _e2938_ = (_e2922_ - _e2882_);
                    let _e2939_ = (_e2927_ - _e2887_);
                    let _e2940_ = (0.5f * _e2936_);
                    let _e2941_ = (0.5f * _e2937_);
                    let _e2942_ = (0.5f * _e2938_);
                    let _e2943_ = (0.5f * _e2939_);
                    let _e2944_ = (_e2932_ * _e2932_);
                    let _e2945_ = (_e2933_ * _e2933_);
                    let _e2946_ = (_e2934_ * _e2934_);
                    let _e2947_ = (_e2935_ * _e2935_);
                    let _e2948_ = (_e2940_ * _e2940_);
                    let _e2949_ = (_e2941_ * _e2941_);
                    let _e2950_ = (_e2942_ * _e2942_);
                    let _e2951_ = (_e2943_ * _e2943_);
                    if (_e2936_ != _e2936_) {
                        phi_10226_ = true;
                    } else {
                        phi_10226_ = (0f >= _e2936_);
                    }
                    let _e2955_ = phi_10226_;
                    let _e2956_ = select(_e2936_, 0f, _e2955_);
                    if (_e2937_ != _e2937_) {
                        phi_10241_ = true;
                    } else {
                        phi_10241_ = (0f >= _e2937_);
                    }
                    let _e2960_ = phi_10241_;
                    let _e2961_ = select(_e2937_, 0f, _e2960_);
                    if (_e2938_ != _e2938_) {
                        phi_10256_ = true;
                    } else {
                        phi_10256_ = (0f >= _e2938_);
                    }
                    let _e2965_ = phi_10256_;
                    let _e2966_ = select(_e2938_, 0f, _e2965_);
                    if (_e2939_ != _e2939_) {
                        phi_10271_ = true;
                    } else {
                        phi_10271_ = (0f >= _e2939_);
                    }
                    let _e2970_ = phi_10271_;
                    let _e2971_ = select(_e2939_, 0f, _e2970_);
                    let _e2972_ = (_e2835_ - _e2816_);
                    let _e2973_ = (_e2839_ - _e2821_);
                    let _e2974_ = (_e2843_ - _e2826_);
                    let _e2975_ = (_e2847_ - _e2831_);
                    if _e2888_ {
                        phi_10286_ = true;
                    } else {
                        phi_10286_ = (-1f <= _e2835_);
                    }
                    let _e2978_ = phi_10286_;
                    if _e2893_ {
                        phi_10301_ = true;
                    } else {
                        phi_10301_ = (-1f <= _e2839_);
                    }
                    let _e2982_ = phi_10301_;
                    if _e2898_ {
                        phi_10316_ = true;
                    } else {
                        phi_10316_ = (-1f <= _e2843_);
                    }
                    let _e2986_ = phi_10316_;
                    if _e2903_ {
                        phi_10331_ = true;
                    } else {
                        phi_10331_ = (-1f <= _e2847_);
                    }
                    let _e2990_ = phi_10331_;
                    let _e2992_ = (select(_e2835_, -1f, _e2978_) - _e2816_);
                    let _e2993_ = (select(_e2839_, -1f, _e2982_) - _e2821_);
                    let _e2994_ = (select(_e2843_, -1f, _e2986_) - _e2826_);
                    let _e2995_ = (select(_e2847_, -1f, _e2990_) - _e2831_);
                    if (_e2992_ != _e2992_) {
                        phi_10346_ = true;
                    } else {
                        phi_10346_ = (0f >= _e2992_);
                    }
                    let _e2999_ = phi_10346_;
                    if (_e2993_ != _e2993_) {
                        phi_10361_ = true;
                    } else {
                        phi_10361_ = (0f >= _e2993_);
                    }
                    let _e3004_ = phi_10361_;
                    if (_e2994_ != _e2994_) {
                        phi_10376_ = true;
                    } else {
                        phi_10376_ = (0f >= _e2994_);
                    }
                    let _e3009_ = phi_10376_;
                    if (_e2995_ != _e2995_) {
                        phi_10391_ = true;
                    } else {
                        phi_10391_ = (0f >= _e2995_);
                    }
                    let _e3014_ = phi_10391_;
                    if _e2848_ {
                        phi_10406_ = true;
                    } else {
                        phi_10406_ = (1f >= _e2816_);
                    }
                    let _e3018_ = phi_10406_;
                    if _e2853_ {
                        phi_10421_ = true;
                    } else {
                        phi_10421_ = (1f >= _e2821_);
                    }
                    let _e3022_ = phi_10421_;
                    if _e2858_ {
                        phi_10436_ = true;
                    } else {
                        phi_10436_ = (1f >= _e2826_);
                    }
                    let _e3026_ = phi_10436_;
                    if _e2863_ {
                        phi_10451_ = true;
                    } else {
                        phi_10451_ = (1f >= _e2831_);
                    }
                    let _e3030_ = phi_10451_;
                    let _e3032_ = (_e2835_ - select(_e2816_, 1f, _e3018_));
                    let _e3033_ = (_e2839_ - select(_e2821_, 1f, _e3022_));
                    let _e3034_ = (_e2843_ - select(_e2826_, 1f, _e3026_));
                    let _e3035_ = (_e2847_ - select(_e2831_, 1f, _e3030_));
                    if (_e3032_ != _e3032_) {
                        phi_10466_ = true;
                    } else {
                        phi_10466_ = (0f >= _e3032_);
                    }
                    let _e3039_ = phi_10466_;
                    if (_e3033_ != _e3033_) {
                        phi_10481_ = true;
                    } else {
                        phi_10481_ = (0f >= _e3033_);
                    }
                    let _e3044_ = phi_10481_;
                    if (_e3034_ != _e3034_) {
                        phi_10496_ = true;
                    } else {
                        phi_10496_ = (0f >= _e3034_);
                    }
                    let _e3049_ = phi_10496_;
                    if (_e3035_ != _e3035_) {
                        phi_10511_ = true;
                    } else {
                        phi_10511_ = (0f >= _e3035_);
                    }
                    let _e3054_ = phi_10511_;
                    let _e3058_ = bitcast<f32>((bitcast<u32>(_e2753_) & 2147483647u));
                    let _e3061_ = bitcast<f32>((bitcast<u32>(_e2758_) & 2147483647u));
                    let _e3064_ = bitcast<f32>((bitcast<u32>(_e2763_) & 2147483647u));
                    let _e3067_ = bitcast<f32>((bitcast<u32>(_e2768_) & 2147483647u));
                    let _e3070_ = bitcast<f32>((bitcast<u32>(_e2795_) & 2147483647u));
                    let _e3073_ = bitcast<f32>((bitcast<u32>(_e2800_) & 2147483647u));
                    let _e3076_ = bitcast<f32>((bitcast<u32>(_e2805_) & 2147483647u));
                    let _e3079_ = bitcast<f32>((bitcast<u32>(_e2810_) & 2147483647u));
                    if (_e3058_ != _e3058_) {
                        phi_10526_ = true;
                    } else {
                        phi_10526_ = (_e3070_ >= _e3058_);
                    }
                    let _e3083_ = phi_10526_;
                    let _e3084_ = select(_e3058_, _e3070_, _e3083_);
                    if (_e3061_ != _e3061_) {
                        phi_10541_ = true;
                    } else {
                        phi_10541_ = (_e3073_ >= _e3061_);
                    }
                    let _e3088_ = phi_10541_;
                    let _e3089_ = select(_e3061_, _e3073_, _e3088_);
                    if (_e3064_ != _e3064_) {
                        phi_10556_ = true;
                    } else {
                        phi_10556_ = (_e3076_ >= _e3064_);
                    }
                    let _e3093_ = phi_10556_;
                    let _e3094_ = select(_e3064_, _e3076_, _e3093_);
                    if (_e3067_ != _e3067_) {
                        phi_10571_ = true;
                    } else {
                        phi_10571_ = (_e3079_ >= _e3067_);
                    }
                    let _e3098_ = phi_10571_;
                    let _e3099_ = select(_e3067_, _e3079_, _e3098_);
                    let _e3116_ = (_e3084_ <= 0f);
                    let _e3117_ = select((_e2932_ * (1.5f - (0.5f * (_e2944_ + _e2948_)))), 0f, _e3116_);
                    let _e3118_ = (_e3089_ <= 0f);
                    let _e3119_ = select((_e2933_ * (1.5f - (0.5f * (_e2945_ + _e2949_)))), 0f, _e3118_);
                    let _e3120_ = (_e3094_ <= 0f);
                    let _e3121_ = select((_e2934_ * (1.5f - (0.5f * (_e2946_ + _e2950_)))), 0f, _e3120_);
                    let _e3122_ = (_e3099_ <= 0f);
                    let _e3123_ = select((_e2935_ * (1.5f - (0.5f * (_e2947_ + _e2951_)))), 0f, _e3122_);
                    let _e3165_ = select(((_e2944_ + (0.33333334f * _e2948_)) - (((_e2944_ * _e2944_) + ((2f * _e2944_) * _e2948_)) + (0.2f * (_e2948_ * _e2948_)))), 0f, _e3116_);
                    let _e3166_ = select(((_e2945_ + (0.33333334f * _e2949_)) - (((_e2945_ * _e2945_) + ((2f * _e2945_) * _e2949_)) + (0.2f * (_e2949_ * _e2949_)))), 0f, _e3118_);
                    let _e3167_ = select(((_e2946_ + (0.33333334f * _e2950_)) - (((_e2946_ * _e2946_) + ((2f * _e2946_) * _e2950_)) + (0.2f * (_e2950_ * _e2950_)))), 0f, _e3120_);
                    let _e3168_ = select(((_e2947_ + (0.33333334f * _e2951_)) - (((_e2947_ * _e2947_) + ((2f * _e2947_) * _e2951_)) + (0.2f * (_e2951_ * _e2951_)))), 0f, _e3122_);
                    if (_e3084_ <= 1f) {
                        if (_e3089_ <= 1f) {
                            let _e3172_ = (_e3094_ <= 1f);
                            if _e3172_ {
                                phi_6424_ = (_e3099_ <= 1f);
                            } else {
                                phi_6424_ = bool();
                            }
                            let _e3175_ = phi_6424_;
                            phi_6426_ = _e3175_;
                            phi_6427_ = select(true, false, _e3172_);
                        } else {
                            phi_6426_ = bool();
                            phi_6427_ = true;
                        }
                        let _e3178_ = phi_6426_;
                        let _e3180_ = phi_6427_;
                        phi_6428_ = _e3178_;
                        phi_6429_ = _e3180_;
                    } else {
                        phi_6428_ = bool();
                        phi_6429_ = true;
                    }
                    let _e3182_ = phi_6428_;
                    let _e3184_ = phi_6429_;
                    if select(_e3182_, false, _e3184_) {
                        phi_6807_ = type_27_(type_22_(array<f32, 4>(_e3117_, _e3119_, _e3121_, _e3123_)), type_22_(array<f32, 4>(_e3165_, _e3166_, _e3167_, _e3168_)), type_22_(_e2811_));
                    } else {
                        if (_e2972_ != _e2972_) {
                            phi_10586_ = true;
                        } else {
                            phi_10586_ = (0.000001f >= _e2972_);
                        }
                        let _e3189_ = phi_10586_;
                        let _e3190_ = select(_e2972_, 0.000001f, _e3189_);
                        if (_e2973_ != _e2973_) {
                            phi_10601_ = true;
                        } else {
                            phi_10601_ = (0.000001f >= _e2973_);
                        }
                        let _e3194_ = phi_10601_;
                        let _e3195_ = select(_e2973_, 0.000001f, _e3194_);
                        if (_e2974_ != _e2974_) {
                            phi_10616_ = true;
                        } else {
                            phi_10616_ = (0.000001f >= _e2974_);
                        }
                        let _e3199_ = phi_10616_;
                        let _e3200_ = select(_e2974_, 0.000001f, _e3199_);
                        if (_e2975_ != _e2975_) {
                            phi_10631_ = true;
                        } else {
                            phi_10631_ = (0.000001f >= _e2975_);
                        }
                        let _e3204_ = phi_10631_;
                        let _e3205_ = select(_e2975_, 0.000001f, _e3204_);
                        let _e3208_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3190_)));
                        let _e3211_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3195_)));
                        let _e3214_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3200_)));
                        let _e3217_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3205_)));
                        let _e3227_ = (_e3208_ + (_e3208_ * (1f - (_e3190_ * _e3208_))));
                        let _e3229_ = (_e3211_ + (_e3211_ * (1f - (_e3195_ * _e3211_))));
                        let _e3231_ = (_e3214_ + (_e3214_ * (1f - (_e3200_ * _e3214_))));
                        let _e3233_ = (_e3217_ + (_e3217_ * (1f - (_e3205_ * _e3217_))));
                        let _e3243_ = (_e3227_ + (_e3227_ * (1f - (_e3190_ * _e3227_))));
                        let _e3245_ = (_e3229_ + (_e3229_ * (1f - (_e3195_ * _e3229_))));
                        let _e3247_ = (_e3231_ + (_e3231_ * (1f - (_e3200_ * _e3231_))));
                        let _e3249_ = (_e3233_ + (_e3233_ * (1f - (_e3205_ * _e3233_))));
                        let _e3302_ = (select((select(0f, 1f, (-1f <= _e2816_)) * select(0f, 1f, (_e2835_ <= 1f))), 1f, (_e2972_ <= 0.000001f)) <= 0f);
                        let _e3304_ = (select((select(0f, 1f, (-1f <= _e2821_)) * select(0f, 1f, (_e2839_ <= 1f))), 1f, (_e2973_ <= 0.000001f)) <= 0f);
                        let _e3306_ = (select((select(0f, 1f, (-1f <= _e2826_)) * select(0f, 1f, (_e2843_ <= 1f))), 1f, (_e2974_ <= 0.000001f)) <= 0f);
                        let _e3308_ = (select((select(0f, 1f, (-1f <= _e2831_)) * select(0f, 1f, (_e2847_ <= 1f))), 1f, (_e2975_ <= 0.000001f)) <= 0f);
                        let _e3314_ = (1f <= _e2816_);
                        let _e3316_ = (1f <= _e2821_);
                        let _e3318_ = (1f <= _e2826_);
                        let _e3320_ = (1f <= _e2831_);
                        let _e3322_ = (_e2835_ <= -1f);
                        let _e3324_ = (_e2839_ <= -1f);
                        let _e3326_ = (_e2843_ <= -1f);
                        let _e3328_ = (_e2847_ <= -1f);
                        phi_6807_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3117_, (((select(_e3032_, 0f, _e3039_) - select(_e2992_, 0f, _e2999_)) + (_e2956_ * _e3117_)) * _e3243_), _e3302_), 1f, _e3314_), -1f, _e3322_), select(select(select(_e3119_, (((select(_e3033_, 0f, _e3044_) - select(_e2993_, 0f, _e3004_)) + (_e2961_ * _e3119_)) * _e3245_), _e3304_), 1f, _e3316_), -1f, _e3324_), select(select(select(_e3121_, (((select(_e3034_, 0f, _e3049_) - select(_e2994_, 0f, _e3009_)) + (_e2966_ * _e3121_)) * _e3247_), _e3306_), 1f, _e3318_), -1f, _e3326_), select(select(select(_e3123_, (((select(_e3035_, 0f, _e3054_) - select(_e2995_, 0f, _e3014_)) + (_e2971_ * _e3123_)) * _e3249_), _e3308_), 1f, _e3320_), -1f, _e3328_))), type_22_(array<f32, 4>(select(select(select(_e3165_, ((_e2956_ * _e3165_) * _e3243_), _e3302_), 0f, _e3314_), 0f, _e3322_), select(select(select(_e3166_, ((_e2961_ * _e3166_) * _e3245_), _e3304_), 0f, _e3316_), 0f, _e3324_), select(select(select(_e3167_, ((_e2966_ * _e3167_) * _e3247_), _e3306_), 0f, _e3318_), 0f, _e3326_), select(select(select(_e3168_, ((_e2971_ * _e3168_) * _e3249_), _e3308_), 0f, _e3320_), 0f, _e3328_))), type_22_(_e2811_));
                    }
                    let _e3349_ = phi_6807_;
                    let _e3356_ = select(0f, _e2726_, (bitcast<f32>((bitcast<u32>(_e2726_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3357_ = (_e3356_ != _e3356_);
                    if _e3357_ {
                        phi_10646_ = true;
                    } else {
                        phi_10646_ = (-1000000000000000000f >= _e3356_);
                    }
                    let _e3360_ = phi_10646_;
                    let _e3361_ = select(_e3356_, -1000000000000000000f, _e3360_);
                    if _e3357_ {
                        phi_10661_ = true;
                    } else {
                        phi_10661_ = (-1000000000000000000f >= _e3356_);
                    }
                    let _e3364_ = phi_10661_;
                    let _e3365_ = select(_e3356_, -1000000000000000000f, _e3364_);
                    if _e3357_ {
                        phi_10676_ = true;
                    } else {
                        phi_10676_ = (-1000000000000000000f >= _e3356_);
                    }
                    let _e3368_ = phi_10676_;
                    let _e3369_ = select(_e3356_, -1000000000000000000f, _e3368_);
                    if _e3357_ {
                        phi_10691_ = true;
                    } else {
                        phi_10691_ = (-1000000000000000000f >= _e3356_);
                    }
                    let _e3372_ = phi_10691_;
                    let _e3373_ = select(_e3356_, -1000000000000000000f, _e3372_);
                    if (_e3361_ != _e3361_) {
                        phi_10706_ = true;
                    } else {
                        phi_10706_ = (1000000000000000000f <= _e3361_);
                    }
                    let _e3377_ = phi_10706_;
                    let _e3378_ = select(_e3361_, 1000000000000000000f, _e3377_);
                    if (_e3365_ != _e3365_) {
                        phi_10721_ = true;
                    } else {
                        phi_10721_ = (1000000000000000000f <= _e3365_);
                    }
                    let _e3382_ = phi_10721_;
                    let _e3383_ = select(_e3365_, 1000000000000000000f, _e3382_);
                    if (_e3369_ != _e3369_) {
                        phi_10736_ = true;
                    } else {
                        phi_10736_ = (1000000000000000000f <= _e3369_);
                    }
                    let _e3387_ = phi_10736_;
                    let _e3388_ = select(_e3369_, 1000000000000000000f, _e3387_);
                    if (_e3373_ != _e3373_) {
                        phi_10751_ = true;
                    } else {
                        phi_10751_ = (1000000000000000000f <= _e3373_);
                    }
                    let _e3392_ = phi_10751_;
                    let _e3393_ = select(_e3373_, 1000000000000000000f, _e3392_);
                    let _e3398_ = select(0f, _e2701_, (bitcast<f32>((bitcast<u32>(_e2701_) & 2147483647u)) <= 340282350000000000000000000000000000000f));
                    let _e3399_ = (_e3398_ != _e3398_);
                    if _e3399_ {
                        phi_10766_ = true;
                    } else {
                        phi_10766_ = (-1000000000000000000f >= _e3398_);
                    }
                    let _e3402_ = phi_10766_;
                    let _e3403_ = select(_e3398_, -1000000000000000000f, _e3402_);
                    if _e3399_ {
                        phi_10781_ = true;
                    } else {
                        phi_10781_ = (-1000000000000000000f >= _e3398_);
                    }
                    let _e3406_ = phi_10781_;
                    let _e3407_ = select(_e3398_, -1000000000000000000f, _e3406_);
                    if _e3399_ {
                        phi_10796_ = true;
                    } else {
                        phi_10796_ = (-1000000000000000000f >= _e3398_);
                    }
                    let _e3410_ = phi_10796_;
                    let _e3411_ = select(_e3398_, -1000000000000000000f, _e3410_);
                    if _e3399_ {
                        phi_10811_ = true;
                    } else {
                        phi_10811_ = (-1000000000000000000f >= _e3398_);
                    }
                    let _e3414_ = phi_10811_;
                    let _e3415_ = select(_e3398_, -1000000000000000000f, _e3414_);
                    if (_e3403_ != _e3403_) {
                        phi_10826_ = true;
                    } else {
                        phi_10826_ = (1000000000000000000f <= _e3403_);
                    }
                    let _e3419_ = phi_10826_;
                    let _e3420_ = select(_e3403_, 1000000000000000000f, _e3419_);
                    if (_e3407_ != _e3407_) {
                        phi_10841_ = true;
                    } else {
                        phi_10841_ = (1000000000000000000f <= _e3407_);
                    }
                    let _e3424_ = phi_10841_;
                    let _e3425_ = select(_e3407_, 1000000000000000000f, _e3424_);
                    if (_e3411_ != _e3411_) {
                        phi_10856_ = true;
                    } else {
                        phi_10856_ = (1000000000000000000f <= _e3411_);
                    }
                    let _e3429_ = phi_10856_;
                    let _e3430_ = select(_e3411_, 1000000000000000000f, _e3429_);
                    if (_e3415_ != _e3415_) {
                        phi_10871_ = true;
                    } else {
                        phi_10871_ = (1000000000000000000f <= _e3415_);
                    }
                    let _e3434_ = phi_10871_;
                    let _e3435_ = select(_e3415_, 1000000000000000000f, _e3434_);
                    let _e3436_ = array<f32, 4>(_e3420_, _e3425_, _e3430_, _e3435_);
                    let _e3437_ = (_e3378_ != _e3378_);
                    if _e3437_ {
                        phi_10886_ = true;
                    } else {
                        phi_10886_ = (_e3420_ <= _e3378_);
                    }
                    let _e3440_ = phi_10886_;
                    let _e3441_ = select(_e3378_, _e3420_, _e3440_);
                    let _e3442_ = (_e3383_ != _e3383_);
                    if _e3442_ {
                        phi_10901_ = true;
                    } else {
                        phi_10901_ = (_e3425_ <= _e3383_);
                    }
                    let _e3445_ = phi_10901_;
                    let _e3446_ = select(_e3383_, _e3425_, _e3445_);
                    let _e3447_ = (_e3388_ != _e3388_);
                    if _e3447_ {
                        phi_10916_ = true;
                    } else {
                        phi_10916_ = (_e3430_ <= _e3388_);
                    }
                    let _e3450_ = phi_10916_;
                    let _e3451_ = select(_e3388_, _e3430_, _e3450_);
                    let _e3452_ = (_e3393_ != _e3393_);
                    if _e3452_ {
                        phi_10931_ = true;
                    } else {
                        phi_10931_ = (_e3435_ <= _e3393_);
                    }
                    let _e3455_ = phi_10931_;
                    let _e3456_ = select(_e3393_, _e3435_, _e3455_);
                    if _e3437_ {
                        phi_10946_ = true;
                    } else {
                        phi_10946_ = (_e3420_ >= _e3378_);
                    }
                    let _e3459_ = phi_10946_;
                    let _e3460_ = select(_e3378_, _e3420_, _e3459_);
                    if _e3442_ {
                        phi_10961_ = true;
                    } else {
                        phi_10961_ = (_e3425_ >= _e3383_);
                    }
                    let _e3463_ = phi_10961_;
                    let _e3464_ = select(_e3383_, _e3425_, _e3463_);
                    if _e3447_ {
                        phi_10976_ = true;
                    } else {
                        phi_10976_ = (_e3430_ >= _e3388_);
                    }
                    let _e3467_ = phi_10976_;
                    let _e3468_ = select(_e3388_, _e3430_, _e3467_);
                    if _e3452_ {
                        phi_10991_ = true;
                    } else {
                        phi_10991_ = (_e3435_ >= _e3393_);
                    }
                    let _e3471_ = phi_10991_;
                    let _e3472_ = select(_e3393_, _e3435_, _e3471_);
                    let _e3473_ = (_e3441_ != _e3441_);
                    if _e3473_ {
                        phi_11006_ = true;
                    } else {
                        phi_11006_ = (-1f >= _e3441_);
                    }
                    let _e3476_ = phi_11006_;
                    let _e3477_ = select(_e3441_, -1f, _e3476_);
                    let _e3478_ = (_e3446_ != _e3446_);
                    if _e3478_ {
                        phi_11021_ = true;
                    } else {
                        phi_11021_ = (-1f >= _e3446_);
                    }
                    let _e3481_ = phi_11021_;
                    let _e3482_ = select(_e3446_, -1f, _e3481_);
                    let _e3483_ = (_e3451_ != _e3451_);
                    if _e3483_ {
                        phi_11036_ = true;
                    } else {
                        phi_11036_ = (-1f >= _e3451_);
                    }
                    let _e3486_ = phi_11036_;
                    let _e3487_ = select(_e3451_, -1f, _e3486_);
                    let _e3488_ = (_e3456_ != _e3456_);
                    if _e3488_ {
                        phi_11051_ = true;
                    } else {
                        phi_11051_ = (-1f >= _e3456_);
                    }
                    let _e3491_ = phi_11051_;
                    let _e3492_ = select(_e3456_, -1f, _e3491_);
                    if (_e3477_ != _e3477_) {
                        phi_11066_ = true;
                    } else {
                        phi_11066_ = (1f <= _e3477_);
                    }
                    let _e3496_ = phi_11066_;
                    let _e3497_ = select(_e3477_, 1f, _e3496_);
                    if (_e3482_ != _e3482_) {
                        phi_11081_ = true;
                    } else {
                        phi_11081_ = (1f <= _e3482_);
                    }
                    let _e3501_ = phi_11081_;
                    let _e3502_ = select(_e3482_, 1f, _e3501_);
                    if (_e3487_ != _e3487_) {
                        phi_11096_ = true;
                    } else {
                        phi_11096_ = (1f <= _e3487_);
                    }
                    let _e3506_ = phi_11096_;
                    let _e3507_ = select(_e3487_, 1f, _e3506_);
                    if (_e3492_ != _e3492_) {
                        phi_11111_ = true;
                    } else {
                        phi_11111_ = (1f <= _e3492_);
                    }
                    let _e3511_ = phi_11111_;
                    let _e3512_ = select(_e3492_, 1f, _e3511_);
                    let _e3513_ = (_e3460_ != _e3460_);
                    if _e3513_ {
                        phi_11126_ = true;
                    } else {
                        phi_11126_ = (-1f >= _e3460_);
                    }
                    let _e3516_ = phi_11126_;
                    let _e3517_ = select(_e3460_, -1f, _e3516_);
                    let _e3518_ = (_e3464_ != _e3464_);
                    if _e3518_ {
                        phi_11141_ = true;
                    } else {
                        phi_11141_ = (-1f >= _e3464_);
                    }
                    let _e3521_ = phi_11141_;
                    let _e3522_ = select(_e3464_, -1f, _e3521_);
                    let _e3523_ = (_e3468_ != _e3468_);
                    if _e3523_ {
                        phi_11156_ = true;
                    } else {
                        phi_11156_ = (-1f >= _e3468_);
                    }
                    let _e3526_ = phi_11156_;
                    let _e3527_ = select(_e3468_, -1f, _e3526_);
                    let _e3528_ = (_e3472_ != _e3472_);
                    if _e3528_ {
                        phi_11171_ = true;
                    } else {
                        phi_11171_ = (-1f >= _e3472_);
                    }
                    let _e3531_ = phi_11171_;
                    let _e3532_ = select(_e3472_, -1f, _e3531_);
                    if (_e3517_ != _e3517_) {
                        phi_11186_ = true;
                    } else {
                        phi_11186_ = (1f <= _e3517_);
                    }
                    let _e3536_ = phi_11186_;
                    let _e3537_ = select(_e3517_, 1f, _e3536_);
                    if (_e3522_ != _e3522_) {
                        phi_11201_ = true;
                    } else {
                        phi_11201_ = (1f <= _e3522_);
                    }
                    let _e3541_ = phi_11201_;
                    let _e3542_ = select(_e3522_, 1f, _e3541_);
                    if (_e3527_ != _e3527_) {
                        phi_11216_ = true;
                    } else {
                        phi_11216_ = (1f <= _e3527_);
                    }
                    let _e3546_ = phi_11216_;
                    let _e3547_ = select(_e3527_, 1f, _e3546_);
                    if (_e3532_ != _e3532_) {
                        phi_11231_ = true;
                    } else {
                        phi_11231_ = (1f <= _e3532_);
                    }
                    let _e3551_ = phi_11231_;
                    let _e3552_ = select(_e3532_, 1f, _e3551_);
                    let _e3557_ = (0.5f * (_e3537_ + _e3497_));
                    let _e3558_ = (0.5f * (_e3542_ + _e3502_));
                    let _e3559_ = (0.5f * (_e3547_ + _e3507_));
                    let _e3560_ = (0.5f * (_e3552_ + _e3512_));
                    let _e3561_ = (_e3537_ - _e3497_);
                    let _e3562_ = (_e3542_ - _e3502_);
                    let _e3563_ = (_e3547_ - _e3507_);
                    let _e3564_ = (_e3552_ - _e3512_);
                    let _e3565_ = (0.5f * _e3561_);
                    let _e3566_ = (0.5f * _e3562_);
                    let _e3567_ = (0.5f * _e3563_);
                    let _e3568_ = (0.5f * _e3564_);
                    let _e3569_ = (_e3557_ * _e3557_);
                    let _e3570_ = (_e3558_ * _e3558_);
                    let _e3571_ = (_e3559_ * _e3559_);
                    let _e3572_ = (_e3560_ * _e3560_);
                    let _e3573_ = (_e3565_ * _e3565_);
                    let _e3574_ = (_e3566_ * _e3566_);
                    let _e3575_ = (_e3567_ * _e3567_);
                    let _e3576_ = (_e3568_ * _e3568_);
                    if (_e3561_ != _e3561_) {
                        phi_11246_ = true;
                    } else {
                        phi_11246_ = (0f >= _e3561_);
                    }
                    let _e3580_ = phi_11246_;
                    let _e3581_ = select(_e3561_, 0f, _e3580_);
                    if (_e3562_ != _e3562_) {
                        phi_11261_ = true;
                    } else {
                        phi_11261_ = (0f >= _e3562_);
                    }
                    let _e3585_ = phi_11261_;
                    let _e3586_ = select(_e3562_, 0f, _e3585_);
                    if (_e3563_ != _e3563_) {
                        phi_11276_ = true;
                    } else {
                        phi_11276_ = (0f >= _e3563_);
                    }
                    let _e3590_ = phi_11276_;
                    let _e3591_ = select(_e3563_, 0f, _e3590_);
                    if (_e3564_ != _e3564_) {
                        phi_11291_ = true;
                    } else {
                        phi_11291_ = (0f >= _e3564_);
                    }
                    let _e3595_ = phi_11291_;
                    let _e3596_ = select(_e3564_, 0f, _e3595_);
                    let _e3597_ = (_e3460_ - _e3441_);
                    let _e3598_ = (_e3464_ - _e3446_);
                    let _e3599_ = (_e3468_ - _e3451_);
                    let _e3600_ = (_e3472_ - _e3456_);
                    if _e3513_ {
                        phi_11306_ = true;
                    } else {
                        phi_11306_ = (-1f <= _e3460_);
                    }
                    let _e3603_ = phi_11306_;
                    if _e3518_ {
                        phi_11321_ = true;
                    } else {
                        phi_11321_ = (-1f <= _e3464_);
                    }
                    let _e3607_ = phi_11321_;
                    if _e3523_ {
                        phi_11336_ = true;
                    } else {
                        phi_11336_ = (-1f <= _e3468_);
                    }
                    let _e3611_ = phi_11336_;
                    if _e3528_ {
                        phi_11351_ = true;
                    } else {
                        phi_11351_ = (-1f <= _e3472_);
                    }
                    let _e3615_ = phi_11351_;
                    let _e3617_ = (select(_e3460_, -1f, _e3603_) - _e3441_);
                    let _e3618_ = (select(_e3464_, -1f, _e3607_) - _e3446_);
                    let _e3619_ = (select(_e3468_, -1f, _e3611_) - _e3451_);
                    let _e3620_ = (select(_e3472_, -1f, _e3615_) - _e3456_);
                    if (_e3617_ != _e3617_) {
                        phi_11366_ = true;
                    } else {
                        phi_11366_ = (0f >= _e3617_);
                    }
                    let _e3624_ = phi_11366_;
                    if (_e3618_ != _e3618_) {
                        phi_11381_ = true;
                    } else {
                        phi_11381_ = (0f >= _e3618_);
                    }
                    let _e3629_ = phi_11381_;
                    if (_e3619_ != _e3619_) {
                        phi_11396_ = true;
                    } else {
                        phi_11396_ = (0f >= _e3619_);
                    }
                    let _e3634_ = phi_11396_;
                    if (_e3620_ != _e3620_) {
                        phi_11411_ = true;
                    } else {
                        phi_11411_ = (0f >= _e3620_);
                    }
                    let _e3639_ = phi_11411_;
                    if _e3473_ {
                        phi_11426_ = true;
                    } else {
                        phi_11426_ = (1f >= _e3441_);
                    }
                    let _e3643_ = phi_11426_;
                    if _e3478_ {
                        phi_11441_ = true;
                    } else {
                        phi_11441_ = (1f >= _e3446_);
                    }
                    let _e3647_ = phi_11441_;
                    if _e3483_ {
                        phi_11456_ = true;
                    } else {
                        phi_11456_ = (1f >= _e3451_);
                    }
                    let _e3651_ = phi_11456_;
                    if _e3488_ {
                        phi_11471_ = true;
                    } else {
                        phi_11471_ = (1f >= _e3456_);
                    }
                    let _e3655_ = phi_11471_;
                    let _e3657_ = (_e3460_ - select(_e3441_, 1f, _e3643_));
                    let _e3658_ = (_e3464_ - select(_e3446_, 1f, _e3647_));
                    let _e3659_ = (_e3468_ - select(_e3451_, 1f, _e3651_));
                    let _e3660_ = (_e3472_ - select(_e3456_, 1f, _e3655_));
                    if (_e3657_ != _e3657_) {
                        phi_11486_ = true;
                    } else {
                        phi_11486_ = (0f >= _e3657_);
                    }
                    let _e3664_ = phi_11486_;
                    if (_e3658_ != _e3658_) {
                        phi_11501_ = true;
                    } else {
                        phi_11501_ = (0f >= _e3658_);
                    }
                    let _e3669_ = phi_11501_;
                    if (_e3659_ != _e3659_) {
                        phi_11516_ = true;
                    } else {
                        phi_11516_ = (0f >= _e3659_);
                    }
                    let _e3674_ = phi_11516_;
                    if (_e3660_ != _e3660_) {
                        phi_11531_ = true;
                    } else {
                        phi_11531_ = (0f >= _e3660_);
                    }
                    let _e3679_ = phi_11531_;
                    let _e3683_ = bitcast<f32>((bitcast<u32>(_e3378_) & 2147483647u));
                    let _e3686_ = bitcast<f32>((bitcast<u32>(_e3383_) & 2147483647u));
                    let _e3689_ = bitcast<f32>((bitcast<u32>(_e3388_) & 2147483647u));
                    let _e3692_ = bitcast<f32>((bitcast<u32>(_e3393_) & 2147483647u));
                    let _e3695_ = bitcast<f32>((bitcast<u32>(_e3420_) & 2147483647u));
                    let _e3698_ = bitcast<f32>((bitcast<u32>(_e3425_) & 2147483647u));
                    let _e3701_ = bitcast<f32>((bitcast<u32>(_e3430_) & 2147483647u));
                    let _e3704_ = bitcast<f32>((bitcast<u32>(_e3435_) & 2147483647u));
                    if (_e3683_ != _e3683_) {
                        phi_11546_ = true;
                    } else {
                        phi_11546_ = (_e3695_ >= _e3683_);
                    }
                    let _e3708_ = phi_11546_;
                    let _e3709_ = select(_e3683_, _e3695_, _e3708_);
                    if (_e3686_ != _e3686_) {
                        phi_11561_ = true;
                    } else {
                        phi_11561_ = (_e3698_ >= _e3686_);
                    }
                    let _e3713_ = phi_11561_;
                    let _e3714_ = select(_e3686_, _e3698_, _e3713_);
                    if (_e3689_ != _e3689_) {
                        phi_11576_ = true;
                    } else {
                        phi_11576_ = (_e3701_ >= _e3689_);
                    }
                    let _e3718_ = phi_11576_;
                    let _e3719_ = select(_e3689_, _e3701_, _e3718_);
                    if (_e3692_ != _e3692_) {
                        phi_11591_ = true;
                    } else {
                        phi_11591_ = (_e3704_ >= _e3692_);
                    }
                    let _e3723_ = phi_11591_;
                    let _e3724_ = select(_e3692_, _e3704_, _e3723_);
                    let _e3741_ = (_e3709_ <= 0f);
                    let _e3742_ = select((_e3557_ * (1.5f - (0.5f * (_e3569_ + _e3573_)))), 0f, _e3741_);
                    let _e3743_ = (_e3714_ <= 0f);
                    let _e3744_ = select((_e3558_ * (1.5f - (0.5f * (_e3570_ + _e3574_)))), 0f, _e3743_);
                    let _e3745_ = (_e3719_ <= 0f);
                    let _e3746_ = select((_e3559_ * (1.5f - (0.5f * (_e3571_ + _e3575_)))), 0f, _e3745_);
                    let _e3747_ = (_e3724_ <= 0f);
                    let _e3748_ = select((_e3560_ * (1.5f - (0.5f * (_e3572_ + _e3576_)))), 0f, _e3747_);
                    let _e3790_ = select(((_e3569_ + (0.33333334f * _e3573_)) - (((_e3569_ * _e3569_) + ((2f * _e3569_) * _e3573_)) + (0.2f * (_e3573_ * _e3573_)))), 0f, _e3741_);
                    let _e3791_ = select(((_e3570_ + (0.33333334f * _e3574_)) - (((_e3570_ * _e3570_) + ((2f * _e3570_) * _e3574_)) + (0.2f * (_e3574_ * _e3574_)))), 0f, _e3743_);
                    let _e3792_ = select(((_e3571_ + (0.33333334f * _e3575_)) - (((_e3571_ * _e3571_) + ((2f * _e3571_) * _e3575_)) + (0.2f * (_e3575_ * _e3575_)))), 0f, _e3745_);
                    let _e3793_ = select(((_e3572_ + (0.33333334f * _e3576_)) - (((_e3572_ * _e3572_) + ((2f * _e3572_) * _e3576_)) + (0.2f * (_e3576_ * _e3576_)))), 0f, _e3747_);
                    if (_e3709_ <= 1f) {
                        if (_e3714_ <= 1f) {
                            let _e3797_ = (_e3719_ <= 1f);
                            if _e3797_ {
                                phi_7231_ = (_e3724_ <= 1f);
                            } else {
                                phi_7231_ = bool();
                            }
                            let _e3800_ = phi_7231_;
                            phi_7233_ = _e3800_;
                            phi_7234_ = select(true, false, _e3797_);
                        } else {
                            phi_7233_ = bool();
                            phi_7234_ = true;
                        }
                        let _e3803_ = phi_7233_;
                        let _e3805_ = phi_7234_;
                        phi_7235_ = _e3803_;
                        phi_7236_ = _e3805_;
                    } else {
                        phi_7235_ = bool();
                        phi_7236_ = true;
                    }
                    let _e3807_ = phi_7235_;
                    let _e3809_ = phi_7236_;
                    if select(_e3807_, false, _e3809_) {
                        phi_7614_ = type_27_(type_22_(array<f32, 4>(_e3742_, _e3744_, _e3746_, _e3748_)), type_22_(array<f32, 4>(_e3790_, _e3791_, _e3792_, _e3793_)), type_22_(_e3436_));
                    } else {
                        if (_e3597_ != _e3597_) {
                            phi_11606_ = true;
                        } else {
                            phi_11606_ = (0.000001f >= _e3597_);
                        }
                        let _e3814_ = phi_11606_;
                        let _e3815_ = select(_e3597_, 0.000001f, _e3814_);
                        if (_e3598_ != _e3598_) {
                            phi_11621_ = true;
                        } else {
                            phi_11621_ = (0.000001f >= _e3598_);
                        }
                        let _e3819_ = phi_11621_;
                        let _e3820_ = select(_e3598_, 0.000001f, _e3819_);
                        if (_e3599_ != _e3599_) {
                            phi_11636_ = true;
                        } else {
                            phi_11636_ = (0.000001f >= _e3599_);
                        }
                        let _e3824_ = phi_11636_;
                        let _e3825_ = select(_e3599_, 0.000001f, _e3824_);
                        if (_e3600_ != _e3600_) {
                            phi_11651_ = true;
                        } else {
                            phi_11651_ = (0.000001f >= _e3600_);
                        }
                        let _e3829_ = phi_11651_;
                        let _e3830_ = select(_e3600_, 0.000001f, _e3829_);
                        let _e3833_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3815_)));
                        let _e3836_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3820_)));
                        let _e3839_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3825_)));
                        let _e3842_ = bitcast<f32>((2129859011u - bitcast<u32>(_e3830_)));
                        let _e3852_ = (_e3833_ + (_e3833_ * (1f - (_e3815_ * _e3833_))));
                        let _e3854_ = (_e3836_ + (_e3836_ * (1f - (_e3820_ * _e3836_))));
                        let _e3856_ = (_e3839_ + (_e3839_ * (1f - (_e3825_ * _e3839_))));
                        let _e3858_ = (_e3842_ + (_e3842_ * (1f - (_e3830_ * _e3842_))));
                        let _e3868_ = (_e3852_ + (_e3852_ * (1f - (_e3815_ * _e3852_))));
                        let _e3870_ = (_e3854_ + (_e3854_ * (1f - (_e3820_ * _e3854_))));
                        let _e3872_ = (_e3856_ + (_e3856_ * (1f - (_e3825_ * _e3856_))));
                        let _e3874_ = (_e3858_ + (_e3858_ * (1f - (_e3830_ * _e3858_))));
                        let _e3927_ = (select((select(0f, 1f, (-1f <= _e3441_)) * select(0f, 1f, (_e3460_ <= 1f))), 1f, (_e3597_ <= 0.000001f)) <= 0f);
                        let _e3929_ = (select((select(0f, 1f, (-1f <= _e3446_)) * select(0f, 1f, (_e3464_ <= 1f))), 1f, (_e3598_ <= 0.000001f)) <= 0f);
                        let _e3931_ = (select((select(0f, 1f, (-1f <= _e3451_)) * select(0f, 1f, (_e3468_ <= 1f))), 1f, (_e3599_ <= 0.000001f)) <= 0f);
                        let _e3933_ = (select((select(0f, 1f, (-1f <= _e3456_)) * select(0f, 1f, (_e3472_ <= 1f))), 1f, (_e3600_ <= 0.000001f)) <= 0f);
                        let _e3939_ = (1f <= _e3441_);
                        let _e3941_ = (1f <= _e3446_);
                        let _e3943_ = (1f <= _e3451_);
                        let _e3945_ = (1f <= _e3456_);
                        let _e3947_ = (_e3460_ <= -1f);
                        let _e3949_ = (_e3464_ <= -1f);
                        let _e3951_ = (_e3468_ <= -1f);
                        let _e3953_ = (_e3472_ <= -1f);
                        phi_7614_ = type_27_(type_22_(array<f32, 4>(select(select(select(_e3742_, (((select(_e3657_, 0f, _e3664_) - select(_e3617_, 0f, _e3624_)) + (_e3581_ * _e3742_)) * _e3868_), _e3927_), 1f, _e3939_), -1f, _e3947_), select(select(select(_e3744_, (((select(_e3658_, 0f, _e3669_) - select(_e3618_, 0f, _e3629_)) + (_e3586_ * _e3744_)) * _e3870_), _e3929_), 1f, _e3941_), -1f, _e3949_), select(select(select(_e3746_, (((select(_e3659_, 0f, _e3674_) - select(_e3619_, 0f, _e3634_)) + (_e3591_ * _e3746_)) * _e3872_), _e3931_), 1f, _e3943_), -1f, _e3951_), select(select(select(_e3748_, (((select(_e3660_, 0f, _e3679_) - select(_e3620_, 0f, _e3639_)) + (_e3596_ * _e3748_)) * _e3874_), _e3933_), 1f, _e3945_), -1f, _e3953_))), type_22_(array<f32, 4>(select(select(select(_e3790_, ((_e3581_ * _e3790_) * _e3868_), _e3927_), 0f, _e3939_), 0f, _e3947_), select(select(select(_e3791_, ((_e3586_ * _e3791_) * _e3870_), _e3929_), 0f, _e3941_), 0f, _e3949_), select(select(select(_e3792_, ((_e3591_ * _e3792_) * _e3872_), _e3931_), 0f, _e3943_), 0f, _e3951_), select(select(select(_e3793_, ((_e3596_ * _e3793_) * _e3874_), _e3933_), 0f, _e3945_), 0f, _e3953_))), type_22_(_e3436_));
                    }
                    let _e3974_ = phi_7614_;
                    if _e158_ {
                        phi_7623_ = (_e201_ + _e3974_.member.member[0]);
                        phi_7624_ = (_e203_ + _e3349_.member.member[0]);
                    } else {
                        phi_7623_ = _e201_;
                        phi_7624_ = _e203_;
                    }
                    let _e3982_ = phi_7623_;
                    let _e3984_ = phi_7624_;
                    workgroupBarrier();
                    phi_12413_ = _e2653_;
                    phi_306_ = _e3982_;
                    phi_308_ = _e3984_;
                } else {
                    phi_12413_ = _e197_;
                    phi_306_ = f32();
                    phi_308_ = f32();
                }
                let _e3986_ = phi_12413_;
                let _e3988_ = phi_306_;
                let _e3990_ = phi_308_;
                phi_11972_ = _e3986_;
                phi_302_ = _e216_;
                phi_305_ = _e3988_;
                phi_307_ = _e3990_;
                phi_12412_ = _e3986_;
                if !(_e222_) {
                    break;
                }
            }
            let _e3993_ = phi_12412_;
            if _e3993_ {
                break;
            }
            if _e158_ {
                if (_e117_ == 0u) {
                    phi_7638_ = 0u;
                } else {
                    let _e3995_ = (_e117_ - 1u);
                    phi_7638_ = select(_e3995_, _e160_, (_e160_ < _e3995_));
                }
                let _e3999_ = phi_7638_;
                if (_e3999_ < _e117_) {
                } else {
                    break;
                }
                let _e4004_ = target_audio.member[_e3999_].x;
                let _e4008_ = target_audio.member[_e3999_].y;
                let _e4009_ = bitcast<u32>(_e4004_);
                let _e4010_ = (_e4009_ & 2139095040u);
                if (_e4010_ == 2139095040u) {
                    phi_7661_ = true;
                } else {
                    phi_7661_ = (_e4010_ == 0u);
                }
                let _e4014_ = phi_7661_;
                let _e4017_ = bitcast<u32>(_e4008_);
                let _e4018_ = (_e4017_ & 2139095040u);
                if (_e4018_ == 2139095040u) {
                    phi_7677_ = true;
                } else {
                    phi_7677_ = (_e4018_ == 0u);
                }
                let _e4022_ = phi_7677_;
                phi_7685_ = vec2<f32>(bitcast<f32>(select(_e4009_, 0u, _e4014_)), bitcast<f32>(select(_e4017_, 0u, _e4022_)));
            } else {
                phi_7685_ = vec2<f32>(0f, 0f);
            }
            let _e4027_ = phi_7685_;
            if _e158_ {
                let _e4030_ = local;
                phi_7691_ = (_e4030_ - _e4027_.x);
            } else {
                phi_7691_ = 0f;
            }
            let _e4033_ = phi_7691_;
            if _e158_ {
                let _e4036_ = local_1_;
                phi_7697_ = (_e4036_ - _e4027_.y);
            } else {
                phi_7697_ = 0f;
            }
            let _e4039_ = phi_7697_;
            if _e158_ {
                let _e4043_ = bitcast<u32>(((_e4033_ * _e4033_) + (_e4039_ * _e4039_)));
                let _e4044_ = (_e4043_ & 2139095040u);
                if (_e4044_ == 2139095040u) {
                    phi_7714_ = true;
                } else {
                    phi_7714_ = (_e4044_ == 0u);
                }
                let _e4048_ = phi_7714_;
                phi_7720_ = bitcast<f32>(select(_e4043_, 0u, _e4048_));
            } else {
                phi_7720_ = 0f;
            }
            let _e4052_ = phi_7720_;
            if _e158_ {
                let _e4058_ = bitcast<u32>(((_e4027_.x * _e4027_.x) + (_e4027_.y * _e4027_.y)));
                let _e4059_ = (_e4058_ & 2139095040u);
                if (_e4059_ == 2139095040u) {
                    phi_7741_ = true;
                } else {
                    phi_7741_ = (_e4059_ == 0u);
                }
                let _e4063_ = phi_7741_;
                phi_7747_ = bitcast<f32>(select(_e4058_, 0u, _e4063_));
            } else {
                phi_7747_ = 0f;
            }
            let _e4067_ = phi_7747_;
            let _e4068_ = bitcast<u32>(_e4052_);
            let _e4069_ = (_e4068_ & 2139095040u);
            if (_e4069_ == 2139095040u) {
                phi_7758_ = true;
            } else {
                phi_7758_ = (_e4069_ == 0u);
            }
            let _e4073_ = phi_7758_;
            let _e4076_ = bitcast<u32>(_e4067_);
            let _e4077_ = (_e4076_ & 2139095040u);
            if (_e4077_ == 2139095040u) {
                phi_7774_ = true;
            } else {
                phi_7774_ = (_e4077_ == 0u);
            }
            let _e4081_ = phi_7774_;
            let _e4085_ = select(255u, _e113_, (_e113_ < 255u));
            let _e4086_ = (_e4085_ < 256u);
            if _e4086_ {
            } else {
                break;
            }
            err_scratch[_e4085_] = bitcast<f32>(select(_e4068_, 0u, _e4073_));
            if _e4086_ {
            } else {
                break;
            }
            tgt_scratch[_e4085_] = bitcast<f32>(select(_e4076_, 0u, _e4081_));
            workgroupBarrier();
            phi_7804_ = 128u;
            loop {
                let _e4090_ = phi_7804_;
                let _e4091_ = (_e4090_ > 0u);
                if _e4091_ {
                    if (_e113_ < _e4090_) {
                        let _e4093_ = (_e113_ + _e4090_);
                        let _e4094_ = (_e4093_ < 256u);
                        if _e4094_ {
                        } else {
                            phi_12820_ = true;
                            break;
                        }
                        let _e4096_ = err_scratch[_e4093_];
                        let _e4097_ = (_e113_ < 256u);
                        if _e4097_ {
                        } else {
                            phi_12820_ = true;
                            break;
                        }
                        let _e4099_ = err_scratch[_e113_];
                        err_scratch[_e113_] = (_e4099_ + _e4096_);
                        if _e4094_ {
                        } else {
                            phi_12820_ = true;
                            break;
                        }
                        let _e4102_ = tgt_scratch[_e4093_];
                        if _e4097_ {
                        } else {
                            phi_12820_ = true;
                            break;
                        }
                        let _e4104_ = tgt_scratch[_e113_];
                        tgt_scratch[_e113_] = (_e4104_ + _e4102_);
                    }
                    workgroupBarrier();
                    phi_7805_ = (_e4090_ >> bitcast<u32>(1i));
                } else {
                    phi_7805_ = u32();
                }
                let _e4109_ = phi_7805_;
                phi_7804_ = _e4109_;
                phi_12820_ = _e3993_;
                if !(_e4091_) {
                    break;
                }
            }
            let _e4112_ = phi_12820_;
            if _e4112_ {
                break;
            }
            let _e4114_ = err_scratch[0];
            let _e4115_ = bitcast<u32>(_e4114_);
            let _e4116_ = (_e4115_ & 2139095040u);
            if (_e4116_ == 2139095040u) {
                phi_7865_ = true;
            } else {
                phi_7865_ = (_e4116_ == 0u);
            }
            let _e4120_ = phi_7865_;
            let _e4124_ = tgt_scratch[0];
            let _e4125_ = bitcast<u32>(_e4124_);
            let _e4126_ = (_e4125_ & 2139095040u);
            if (_e4126_ == 2139095040u) {
                phi_7883_ = true;
            } else {
                phi_7883_ = (_e4126_ == 0u);
            }
            let _e4130_ = phi_7883_;
            workgroupBarrier();
            if (_e113_ == 0u) {
                if (_e112_.x < _e129_) {
                    if (_e112_.y < _e149_) {
                        if (_e112_.z < select(1u, _e140_, (1u < _e140_))) {
                            let _e4139_ = uniforms.member.member_5_;
                            if ((bitcast<u32>(_e4139_) & 2139095040u) != 2139095040u) {
                                let _e4143_ = (_e4139_ > 0.000000000001f);
                                phi_7923_ = select(f32(), _e4139_, _e4143_);
                                phi_7924_ = select(true, false, _e4143_);
                            } else {
                                phi_7923_ = f32();
                                phi_7924_ = true;
                            }
                            let _e4147_ = phi_7923_;
                            let _e4149_ = phi_7924_;
                            let _e4151_ = (bitcast<f32>(select(_e4125_, 0u, _e4130_)) + select(_e4147_, 0.000000000001f, _e4149_));
                            if ((bitcast<u32>(_e4151_) & 2139095040u) != 2139095040u) {
                                let _e4155_ = (_e4151_ <= 0f);
                                if _e4155_ {
                                    phi_8022_ = f32();
                                } else {
                                    if (_e4151_ < 0.00001f) {
                                        let _e4187_ = (select(0.000000000000000000000001f, _e4151_, (_e4151_ > 0.000000000000000000000001f)) * 18446744000000000000f);
                                        if (_e4187_ != _e4187_) {
                                            phi_11726_ = true;
                                        } else {
                                            phi_11726_ = (0.000001f >= _e4187_);
                                        }
                                        let _e4191_ = phi_11726_;
                                        let _e4192_ = select(_e4187_, 0.000001f, _e4191_);
                                        let _e4195_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4192_)));
                                        let _e4199_ = (_e4195_ + (_e4195_ * (1f - (_e4192_ * _e4195_))));
                                        let _e4203_ = (_e4199_ + (_e4199_ * (1f - (_e4192_ * _e4199_))));
                                        let _e4209_ = bitcast<u32>(((_e4203_ + (_e4203_ * (1f - (_e4192_ * _e4203_)))) * 18446744000000000000f));
                                        let _e4210_ = (_e4209_ & 2139095040u);
                                        if (_e4210_ == 2139095040u) {
                                            phi_7980_ = true;
                                        } else {
                                            phi_7980_ = (_e4210_ == 0u);
                                        }
                                        let _e4214_ = phi_7980_;
                                        phi_8021_ = bitcast<f32>(select(_e4209_, 0u, _e4214_));
                                    } else {
                                        if (_e4151_ != _e4151_) {
                                            phi_11666_ = true;
                                        } else {
                                            phi_11666_ = (0.000001f >= _e4151_);
                                        }
                                        let _e4160_ = phi_11666_;
                                        let _e4161_ = select(_e4151_, 0.000001f, _e4160_);
                                        let _e4164_ = bitcast<f32>((2129859011u - bitcast<u32>(_e4161_)));
                                        let _e4168_ = (_e4164_ + (_e4164_ * (1f - (_e4161_ * _e4164_))));
                                        let _e4172_ = (_e4168_ + (_e4168_ * (1f - (_e4161_ * _e4168_))));
                                        let _e4177_ = bitcast<u32>((_e4172_ + (_e4172_ * (1f - (_e4161_ * _e4172_)))));
                                        let _e4178_ = (_e4177_ & 2139095040u);
                                        if (_e4178_ == 2139095040u) {
                                            phi_8015_ = true;
                                        } else {
                                            phi_8015_ = (_e4178_ == 0u);
                                        }
                                        let _e4182_ = phi_8015_;
                                        phi_8021_ = bitcast<f32>(select(_e4177_, 0u, _e4182_));
                                    }
                                    let _e4218_ = phi_8021_;
                                    phi_8022_ = _e4218_;
                                }
                                let _e4220_ = phi_8022_;
                                phi_8023_ = _e4220_;
                                phi_8024_ = _e4155_;
                            } else {
                                phi_8023_ = f32();
                                phi_8024_ = true;
                            }
                            let _e4222_ = phi_8023_;
                            let _e4224_ = phi_8024_;
                            let _e4227_ = bitcast<u32>((bitcast<f32>(select(_e4115_, 0u, _e4120_)) * select(_e4222_, 0f, _e4224_)));
                            let _e4228_ = (_e4227_ & 2139095040u);
                            if (_e4228_ == 2139095040u) {
                                phi_8040_ = true;
                            } else {
                                phi_8040_ = (_e4228_ == 0u);
                            }
                            let _e4232_ = phi_8040_;
                            let _e4238_ = ((((_e112_.z * _e129_) + _e112_.x) * _e149_) + _e112_.y);
                            if (_e121_ == 0u) {
                                phi_8060_ = 0u;
                            } else {
                                let _e4240_ = (_e121_ - 1u);
                                phi_8060_ = select(_e4240_, _e4238_, (_e4238_ < _e4240_));
                            }
                            let _e4244_ = phi_8060_;
                            if (_e4244_ < _e121_) {
                            } else {
                                break;
                            }
                            seed_losses.member[_e4244_] = bitcast<f32>(select(_e4227_, 0u, _e4232_));
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
fn prescreen_population(@builtin(workgroup_id) wg_id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    wg_id_1_ = wg_id;
    lid_1_ = lid;
    function_();
    return;
}
